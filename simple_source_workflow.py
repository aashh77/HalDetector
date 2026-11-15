
import math


def clip(x):return max(0.,min(1.,float(x)))
def sigmoid(x):return 1/(1+math.exp(-max(-60.,min(60.,float(x)))))
def source_score(B,margin,cosine,relevance,nli):
    # QA confidence alone saturates; margin retains useful ordering. NLI on an
    # extracted sentence is mostly a fidelity check and deliberately low weight.
    M=clip(margin/15.)
    return .35*B+.25*M+.20*clip(cosine)+.10*clip(relevance)+.10*clip(nli)


def containing_sentence(text,start,end):
    if not (0<=start<end<=len(text)):raise ValueError('Invalid QA offsets')
    left=max(text.rfind('.',0,start),text.rfind('!',0,start),text.rfind('?',0,start))+1
    stops=[i for i in [text.find('.',end),text.find('!',end),text.find('?',end)] if i>=0]
    right=min(stops)+1 if stops else len(text)
    return text[left:right].strip()


def score_source(question,row,models,pairs,args,cfg,provenance):
    item=dict(source_id=row['id'],title=row.get('title'),url=row.get('url'),
        retrieved_at=row.get('retrieved_at'),passage=row['passage'],answer=None,eligible=False,score=None)
    try:
        qa=models.qa(question,row['passage']);answer=qa['answer'];item['answer']=answer
        exact=bool(answer and row['passage'][qa['start']:qa['end']]==answer)
        cosine=float(models.similarity(question,row['passage']))
        R=float(row.get('relevance',0.))
        sentence=containing_sentence(row['passage'],qa['start'],qa['end']) if exact else ''
        # Do not turn a fragment into an invented factual claim. The hypothesis
        # is the verbatim answer-bearing source sentence. This checks fidelity,
        # not whether the sentence answers the question: QA/linkage do that.
        nli2=models.entail(row['passage'],sentence) if sentence else None
        nli1=pairs.entail(row['passage'],sentence) if sentence else None
        E=float(nli2['entailment']) if nli2 else 0.
        B=float(qa['answerability']);margin=float(qa['null_margin'])
        if not all(math.isfinite(v) for v in [B,margin,cosine,R,E]):raise ValueError('Nonfinite source metric')
        score=source_score(B,margin,cosine,R,E)
        gates=dict(exact_answer_span=exact,qa_answerability=B>=cfg['qa_answerability'],
            qa_margin=margin>=cfg['qa_margin'],query_relevance=R>=args.min_relevance,
            semantic_linkage=cosine>=cfg['embedding_relevance'],source_provenance=provenance(row))
        item.update(answer_sentence=sentence,score=score,eligible=all(gates.values()),gates=gates,
            model_outputs={
                'deepset/roberta-base-squad2':qa,
                'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli':dict(
                    premise=row['passage'],hypothesis=sentence,**(nli2 or {}),
                    role='source sentence fidelity diagnostic, NOT question answerability or source truth'),
                'sentence-transformers/all-MiniLM-L6-v2':dict(question_passage_cosine=cosine),
                'existing_query_passage_ranker':dict(relevance=R,logit=row.get('relevance_logit')),
                'existing_nli1':nli1},
            metrics=dict(qa_answerability=B,qa_margin=margin,margin_index=clip(margin/15.),
                exact_span=exact,question_passage_cosine=cosine,query_passage_relevance=R,
                source_sentence_entailment=E),
            score_formula='0.35*QA_answerability+0.25*clip(QA_margin/15)+0.20*clip(question_passage_cosine)+0.10*relevance+0.10*NLI2_sentence_fidelity',
            score_is_calibrated_probability=False)
    except Exception as exc:item['error']=f'{type(exc).__name__}: {exc}'
    return item


def choose(records,threshold):
    eligible=[x for x in records if x.get('eligible') and x.get('score') is not None]
    if not eligible:return None,'no_source_passed_minimum_checks'
    selected=max(eligible,key=lambda x:x['score'])
    if selected['score']<threshold:return selected,'best_source_below_final_threshold'
    return selected,'selected_source_passed'


def answer_match(question,original,selected,models,cfg,agreement):
    qa=models.qa(question,original)
    check=agreement(qa['answer'],selected,models,cfg)
    valid=bool(qa.get('answer_is_exact_context_span') and qa['answerability']>=cfg['qa_answerability']
               and qa['null_margin']>=cfg['qa_margin'])
    return dict(matches=valid and check['equivalent'],original_answer_span=qa['answer'],
        selected_answer_span=selected,original_answer_qa=qa,answer_equivalence=check,
        definition='Compare question-specific extracted answer spans, not full wording or original factual authority.')


def finalize(result,question,evidence,tag,pairs,args,api):
    import generic_verifier as g
    models=g.extra_models();cfg=g.settings()
    if hasattr(api,'DEFINITIONS'):
        api.DEFINITIONS['final_overall']='Best eligible source score; no cross-source consensus/veto. Different answer: evidence score only. Matching answer: 0.80*evidence+0.20*(probe_weight*RAG_probe+(1-probe_weight)*RAG_L). Experimental, not calibrated.'
    result['workflow_version']='single_source_rank_v1'
    result['source_selection_policy']=dict(consensus_required=False,cross_source_contradiction_veto=False,
        warning='Requested best-source policy ignores disagreement from unselected sources. It can select a confidently wrong source; no source-truth guarantee.')
    result['per_source_results']=[score_source(question,row,models,pairs,args,cfg,g.provenance) for row in evidence]
    selected,reason=choose(result['per_source_results'],args.final_threshold)
    result['selected_rag']=dict(answer=selected['answer'] if selected else None,
        source_ids=[selected['source_id']] if selected else [],evidence_score=selected['score'] if selected else None,
        threshold=args.final_threshold,selection_reason=reason,consensus_required=False,
        cross_source_contradiction_veto=False)
    result['rag_candidate']=selected['answer'] if selected else None
    result['rag_probe_diagnostic']=None;result['rag_answer_match']=None
    result['rag_final_scoring']=dict(branch='no_accepted_source',final_score=None,accepted=False)
    result['verification']=dict(accepted=False,reason=reason,overall_score=selected['score'] if selected else None,
        support_ids=[],per_source_results=result['per_source_results'],selection_policy=result['source_selection_policy'])
    result['final_answer']='abstain';result['action']='abstain';result['sources']=[]
    if reason!='selected_source_passed':return result
    answer=selected['answer'];source=next(x for x in evidence if x['id']==selected['source_id'])
    # Score every selected RAG answer under actual evidence context. Failure of
    # this diagnostic never fabricates a score or changes the evidence branch.
    prompt=api.rag_prompt(question,[source],0)
    try:probe=tag.measure(prompt,answer,rag=True)
    except Exception as exc:probe=dict(available=False,error=f'{type(exc).__name__}: {exc}',excluded_from_rag_release=True)
    result['rag_probe_diagnostic']=probe
    try:match=answer_match(question,result.get('original_tag_output',''),answer,models,cfg,g.agreement)
    except Exception as exc:match=dict(matches=False,status='match_unavailable',error=str(exc))
    result['rag_answer_match']=match
    evidence_score=selected['score'];w=float(getattr(args,'probe_weight',.8))
    if match['matches']:
        if probe.get('available'):
            P=float(probe['p_correct']);L=float(probe['token_likelihood'])
            if not all(math.isfinite(v) and 0<=v<=1 for v in [P,L]):raise ValueError('Invalid RAG probe/likelihood score')
            PL=w*P+(1-w)*L
            # Include requested probe+L without reinstating old OOD-probe veto.
            F=.80*evidence_score+.20*PL
            branch='matching_answer_evidence_plus_probe_likelihood'
            accepted=F>=args.final_threshold
            scoring=dict(branch=branch,probe_score=P,likelihood_score=L,probe_likelihood_score=PL,
                evidence_score=evidence_score,final_score=F,threshold=args.final_threshold,accepted=accepted,
                formula='F=0.80*selected_source_score+0.20*(probe_weight*RAG_probe+(1-probe_weight)*RAG_L)',
                caveat='RAG probe is out-of-distribution; this composite is an experimental index, not calibrated confidence.')
        else:
            F=evidence_score;accepted=True
            scoring=dict(branch='matching_answer_probe_unavailable_evidence_only',final_score=F,accepted=True,
                evidence_score=evidence_score,threshold=args.final_threshold,
                note='No probe/L values invented. Evidence-only fallback explicitly reported.')
    else:
        F=evidence_score;accepted=True
        scoring=dict(branch='different_answer_selected_source_evidence_only',evidence_score=evidence_score,
            final_score=F,threshold=args.final_threshold,accepted=True,
            formula='F=selected_source_score; original answer and OOD RAG probe cannot veto a correction.')
    scoring['score_is_calibrated_probability']=False
    result['rag_final_scoring']=scoring
    result['verification'].update(accepted=accepted,reason='final_score_passed' if accepted else 'matching_branch_final_score_below_threshold',
        overall_score=F,support_ids=[source['id']] if accepted else [],
        final_scoring=scoring,definitions=dict(overall=scoring.get('formula','Selected source evidence score')))
    result['final_score']=F
    if accepted:
        result.update(final_answer=answer,action='answer_selected_source',sources=[source],
            final_answer_origin='exact_QA_span_from_highest_scoring_eligible_source')
    return result


def compact_result(full):
    """Ordered JSON view only. The original full object still feeds markdown()."""
    original=full.get('original_probe',{});metrics=full.get('original_metrics',{})
    threshold=full.get('thresholds',{}).get('original')
    overall=metrics.get('overall_score')
    selected=full.get('selected_rag')
    final=full.get('rag_final_scoring',{})
    return dict(
        question=full.get('question'),
        tag_model_output=full.get('original_tag_output'),
        probe_score=dict(value=original.get('p_correct'),noncorrect=original.get('p_not_correct'),
            definition=original.get('definition'),available=original.get('available',False)),
        likelihood_score=dict(value=original.get('token_likelihood'),definition=original.get('token_likelihood_definition')),
        original_overall_score=dict(value=overall,formula=metrics.get('formula'),probe_weight=metrics.get('probe_weight'),
            score_is_calibrated_probability=False),
        threshold_comparison=dict(threshold=threshold,below_threshold=(overall<threshold if overall is not None and threshold is not None else None),
            retrieval_triggered=full.get('retrieval_done',False),reasons=full.get('retrieval_reasons',[])),
        retrieval_sources=full.get('retrieval_sources',[]),
        source_relevance_and_answers=full.get('per_source_results',[]),
        selected_rag_candidate=selected,
        rag_answer_matches_original=full.get('rag_answer_match'),
        rag_probe_score=dict(value=full.get('rag_probe_diagnostic',{}).get('p_correct'),
            available=full.get('rag_probe_diagnostic',{}).get('available',False),
            calibration_scope=full.get('rag_probe_diagnostic',{}).get('calibration_scope'),
            error=full.get('rag_probe_diagnostic',{}).get('error')) if full.get('rag_probe_diagnostic') else None,
        rag_likelihood_score=full.get('rag_probe_diagnostic',{}).get('token_likelihood') if full.get('rag_probe_diagnostic') else None,
        rag_final_scoring=final or None,
        final_score=full.get('final_score',overall if not full.get('retrieval_done') else None),
        final_answer=full.get('final_answer','abstain'),
        final_sources=full.get('sources',[]),
        policy_notice=full.get('source_selection_policy'),
        error=full.get('error'))
