"""Generic factoid-QA verification for TAG RAG v3. No question-specific patterns.
All combined/model scores are experimental indices, not truth probabilities.
"""
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import string
from urllib.parse import urlparse

QA_ID='deepset/roberta-base-squad2'
NLI2_ID='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
EMBED_ID='sentence-transformers/all-MiniLM-L6-v2'
DEFAULTS=dict(qa_answerability=.70,qa_margin=1.0,qa_agreement=.65,semantic_equivalence=.88,
    ensemble_entailment=.65,each_entailment=.40,conflict_consensus=.80,embedding_relevance=.10,
    extra_original_conflict=.70,max_evidence=6,max_answer_words=65)


def normalized(text):return re.sub(r'\s+',' ',str(text)).strip()
def sig(x):return 1/(1+math.exp(-max(-60,min(60,float(x)))))
def answer_tokens(text):
    text=normalized(text).casefold().translate(str.maketrans('','',string.punctuation))
    return [x for x in text.split() if x not in {'a','an','the'}]
def f1(a,b):
    from collections import Counter
    aa=answer_tokens(a);bb=answer_tokens(b)
    if not aa or not bb:return 0.
    overlap=sum((Counter(aa)&Counter(bb)).values())
    return 2*overlap/(len(aa)+len(bb))
def numbers(text):return re.findall(r'\d+(?:\.\d+)?',text)


@lru_cache(maxsize=1)
def settings():
    cfg=dict(DEFAULTS)
    path=os.getenv('TAG_GENERIC_CONFIG')
    if path:
        extra=json.loads(Path(path).read_text())
        if not isinstance(extra,dict) or set(extra)-set(cfg):raise ValueError('Unknown generic verifier config keys')
        cfg.update(extra)
    for k,v in cfg.items():
        if not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Nonfinite config: '+k)
        if k not in {'qa_margin','max_evidence','max_answer_words'} and not 0<=v<=1:raise ValueError('Config outside [0,1]: '+k)
    if not 1<=cfg['max_evidence']<=12 or int(cfg['max_evidence'])!=cfg['max_evidence']:raise ValueError('max_evidence must be integer 1..12')
    if not 4<=cfg['max_answer_words']<=100 or int(cfg['max_answer_words'])!=cfg['max_answer_words']:raise ValueError('max_answer_words must be integer 4..100')
    return cfg


class AdditionalModels:
    """All extra models on CPU; existing TAG/probe environment is left unchanged."""
    def __init__(self):
        import torch
        from transformers import AutoTokenizer,AutoModel,AutoModelForQuestionAnswering,AutoModelForSequenceClassification
        self.torch=torch
        self.qt=AutoTokenizer.from_pretrained(QA_ID,revision=os.getenv('TAG_QA_REVISION','main'),use_fast=True)
        self.qm=AutoModelForQuestionAnswering.from_pretrained(QA_ID,revision=os.getenv('TAG_QA_REVISION','main')).cpu().eval()
        self.nt=AutoTokenizer.from_pretrained(NLI2_ID,revision=os.getenv('TAG_NLI2_REVISION','main'))
        self.nm=AutoModelForSequenceClassification.from_pretrained(NLI2_ID,revision=os.getenv('TAG_NLI2_REVISION','main')).cpu().eval()
        labels={int(k):str(v).lower() for k,v in self.nm.config.id2label.items()}
        # Model card: entailment, neutral, contradiction. Fail on incompatible config.
        if self.nm.config.num_labels!=3:raise ValueError('Expected three NLI2 labels')
        self.e=next((i for i,v in labels.items() if 'entail' in v),0)
        self.c=next((i for i,v in labels.items() if 'contrad' in v),2)
        self.et=AutoTokenizer.from_pretrained(EMBED_ID,revision=os.getenv('TAG_EMBED_REVISION','main'))
        self.em=AutoModel.from_pretrained(EMBED_ID,revision=os.getenv('TAG_EMBED_REVISION','main')).cpu().eval()
        self.qa_cache={};self.emb_cache={};self.nli_cache={}
        self.versions={name:dict(model=mid,revision=getattr(model.config,'_commit_hash',None))
            for name,mid,model in [('qa',QA_ID,self.qm),('nli2',NLI2_ID,self.nm),('embedding',EMBED_ID,self.em)]}
    def qa(self,question,context):
        key=(question,context)
        if key in self.qa_cache:return self.qa_cache[key]
        torch=self.torch
        batch=self.qt(question,context,truncation='only_second',max_length=384,stride=96,
            return_overflowing_tokens=True,return_offsets_mapping=True,padding=True,return_tensors='pt')
        offsets=batch.pop('offset_mapping');batch.pop('overflow_to_sample_mapping',None)
        with torch.inference_mode():out=self.qm(**batch)
        best=None
        for w in range(batch['input_ids'].shape[0]):
            seq=batch.sequence_ids(w)
            valid=[i for i,s in enumerate(seq) if s==1 and int(offsets[w,i,1])>int(offsets[w,i,0])]
            if not valid:continue
            starts=out.start_logits[w].float();ends=out.end_logits[w].float()
            cls=(batch['input_ids'][w]==self.qt.cls_token_id).nonzero(as_tuple=True)[0]
            if len(cls)==0:raise ValueError('QA CLS token missing')
            null=float(starts[int(cls[0])]+ends[int(cls[0])])
            top_s=sorted(valid,key=lambda i:float(starts[i]),reverse=True)[:20]
            top_e=sorted(valid,key=lambda i:float(ends[i]),reverse=True)[:20]
            validset=set(valid)
            for i in top_s:
                for j in top_e:
                    if j<i or j-i+1>30 or any(t not in validset for t in range(i,j+1)):continue
                    start=int(offsets[w,i,0]);end=int(offsets[w,j,1]);text=context[start:end]
                    margin=float(starts[i]+ends[j])-null
                    item=dict(answer=text,start=start,end=end,null_margin=margin,answerability=sig(margin),
                        answer_is_exact_context_span=context[start:end]==text,
                        note='sigmoid(best span logit sum minus CLS null logit sum); uncalibrated answerability index')
                    if best is None or margin>best['null_margin']:best=item
        result=best or dict(answer='',start=0,end=0,null_margin=-60.,answerability=0.,answer_is_exact_context_span=False)
        self.qa_cache[key]=result;return result
    def embedding(self,text):
        if text not in self.emb_cache:
            b=self.et(text,return_tensors='pt',truncation=True,max_length=256)
            with self.torch.inference_mode():h=self.em(**b).last_hidden_state
            mask=b['attention_mask'].unsqueeze(-1).float();v=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
            v=self.torch.nn.functional.normalize(v,p=2,dim=1)[0]
            self.emb_cache[text]=v
        return self.emb_cache[text]
    def similarity(self,a,b):return float((self.embedding(a)*self.embedding(b)).sum())
    def entail(self,premise,hypothesis):
        key=(premise,hypothesis)
        if key not in self.nli_cache:
            b=self.nt(premise,hypothesis,return_tensors='pt',truncation=False)
            if b['input_ids'].shape[1]>512:raise ValueError('NLI2 context >512 tokens; not silently truncated')
            with self.torch.inference_mode():p=self.nm(**b).logits.float().softmax(-1)[0].tolist()
            self.nli_cache[key]=dict(entailment=float(p[self.e]),contradiction=float(p[self.c]))
        return self.nli_cache[key]


@lru_cache(maxsize=1)
def extra_models():return AdditionalModels()


def provenance(row):
    u=urlparse(str(row.get('url','')).strip())
    return bool(u.scheme=='https' and u.hostname=='en.wikipedia.org' and row.get('pageid')
                and row.get('revision_id') and row.get('retrieved_at'))


def agreement(a,b,models,cfg):
    lexical=f1(a,b);cos=models.similarity(a,b) if a and b else 0.
    # Similar names/dates alone must not pass semantic equivalence.
    num_ok=numbers(a)==numbers(b)
    equivalent=lexical>=cfg['qa_agreement'] and num_ok
    semantic=None
    if not equivalent and cos>=cfg['semantic_equivalence'] and num_ok:
        ab=models.entail(a,b);ba=models.entail(b,a)
        semantic=dict(forward=ab,reverse=ba)
        equivalent=min(ab['entailment'],ba['entailment'])>=.85
    score=max(lexical,min(.95,cos) if equivalent and semantic else 0.) if num_ok else 0.
    return dict(score=score,lexical_f1=lexical,embedding_cosine=cos,equivalent=equivalent,
                numeric_consistency=num_ok,semantic_equivalence_checks=semantic)


def verify(question,answer,evidence,pairs,args):
    """Generic replacement: never calls old actor/river-pattern wrappers."""
    cfg=settings();models=extra_models()
    if not 0<=args.final_threshold<=1:raise ValueError('final_threshold outside [0,1]')
    if not answer.strip():return dict(accepted=False,reason='empty_answer',overall_score=0.,support_ids=[])
    if len(answer.split())>cfg['max_answer_words']:
        return dict(accepted=False,reason='answer_exceeds_factoid_scope',overall_score=0.,support_ids=[])
    candidate_qa=models.qa(question,answer)
    records=[];support=[];conflicts=[]
    evidence=evidence[:int(cfg['max_evidence'])]
    if len({x['id'] for x in evidence})!=len(evidence):raise ValueError('Duplicate evidence source IDs')
    for row in evidence[:int(cfg['max_evidence'])]:
        record=dict(source_id=row['id'],relevance=float(row.get('relevance',0.)))
        try:
            qa=models.qa(question,row['passage']);cos=models.similarity(question,row['passage'])
            answerable=bool(qa['answer'] and qa['answer_is_exact_context_span'] and
                qa['answerability']>=cfg['qa_answerability'] and qa['null_margin']>=cfg['qa_margin'])
            applicable=answerable and record['relevance']>=args.min_relevance and cos>=cfg['embedding_relevance']
            record.update(qa=qa,question_passage_cosine=cos,applicable=applicable,
                applicability_reason='answerable_and_relevant' if applicable else 'no_answer_or_insufficient_question_linkage')
            if not applicable:records.append(record);continue
            match=agreement(candidate_qa['answer'],qa['answer'],models,cfg)
            first=pairs.entail(row['passage'],answer);second=models.entail(row['passage'],answer)
            ensemble=(first['entailment']+second['entailment'])/2
            consensus_c=min(first['contradiction'],second['contradiction'])
            candidate_answerable=candidate_qa['answerability']>=cfg['qa_answerability']
            supports=bool(match['equivalent'] and candidate_answerable and
                ensemble>=max(args.min_entailment,cfg['ensemble_entailment']) and
                min(first['entailment'],second['entailment'])>=cfg['each_entailment'] and provenance(row))
            # Conflict requires answerable evidence with a DIFFERENT extracted answer,
            # plus BOTH NLI models agreeing on contradiction. Omission alone cannot veto.
            conflict=bool(not match['equivalent'] and consensus_c>=cfg['conflict_consensus'])
            record.update(answer_agreement=match,nli1=first,nli2=second,ensemble_entailment=ensemble,
                consensus_contradiction=consensus_c,supports=supports,conflicts=conflict,provenance=provenance(row))
            if supports:support.append(record)
            if conflict:conflicts.append(record)
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}',applicable=False)
        records.append(record)
    if support:
        best=max(support,key=lambda x:.45*x['ensemble_entailment']+.25*x['answer_agreement']['score']+
                 .15*x['relevance']+.15*x['qa']['answerability'])
        E=best['ensemble_entailment'];Q=best['answer_agreement']['score'];R=best['relevance'];B=best['qa']['answerability']
        F=.45*E+.25*Q+.15*R+.15*B
    else:best=None;E=Q=R=B=F=0.
    C=max([x['consensus_contradiction'] for x in conflicts],default=0.)
    errors=[x for x in records if x.get('error')]
    gates=dict(has_qa_and_dual_nli_support=bool(support),no_verified_competing_answer=not conflicts,
        overall=F>=args.final_threshold,provenance=bool(best and best['provenance']),
        no_unresolved_verifier_errors=not errors)
    return dict(accepted=all(gates.values()),reason='generic_qa_dual_nli_passed' if all(gates.values()) else 'generic_verification_gate_failed',
        overall_score=F,entailment=E,qa_answer_agreement=Q,supporting_relevance=R,qa_answerability=B,
        contradiction=C,gates=gates,support_ids=[x['source_id'] for x in support],
        candidate_qa=candidate_qa,passage_checks=records,competing_answer_sources=[x['source_id'] for x in conflicts],
        verifier_models=models.versions,settings=cfg,score_is_calibrated_probability=False,
        evidence_truth_probability=None,
        definitions=dict(overall='F=.45*mean(NLI1,NLI2 entailment)+.25*QA answer agreement+.15*relevance+.15*QA answerability, from strongest qualifying support; every hard gate must pass.',
            applicability='Question-answerable extractive span + query-passage relevance + embedding cosine, independent of original answer.',
            contradiction='Only competing QA answers with BOTH NLI models above conflict threshold can veto; unrelated/answerless passages are not contradictions.',
            qa_agreement='Token F1 of independently extracted question answers, or high embedding cosine plus bidirectional NLI equivalence. Not full-answer embedding similarity.',
            scope='Experimental concise English factoid QA; not arbitrary synthesis, multi-hop or source-truth verification.'))


def comparison(original,candidate,pairs,models,cfg):
    # Original is not evidence. Disagreement triggers MORE checking, never a veto by itself.
    a=pairs.entail(original,candidate);b=pairs.entail(candidate,original)
    c=models.entail(original,candidate);d=models.entail(candidate,original)
    disagreement=max(a['contradiction'],b['contradiction'],c['contradiction'],d['contradiction'])>=.5
    return dict(original_to_candidate_nli1=a,candidate_to_original_nli1=b,
                original_to_candidate_nli2=c,candidate_to_original_nli2=d,
                contradiction_detected=disagreement,original_used_as_ground_truth=False)


def finalize(result,question,evidence,tag,pairs,args,api):
    """Called after retrieval; replaces old pattern recovery and TAG-only attempts."""
    models=extra_models();cfg=settings()
    result['generic_verifier_version']='qa_dual_nli_v1'
    result['generic_model_versions']=models.versions
    result['generic_thresholds']=dict(settings=cfg,final_threshold=args.final_threshold,min_relevance=args.min_relevance,min_entailment=args.min_entailment,
        legacy_gate_note='Old min_answer_relevance and max_contradiction gates are replaced by QA linkage and dual-NLI verified competing-answer checks. Original routing thresholds are unchanged.')
    result['verification_evidence_ids']=[x['id'] for x in evidence]
    result['evidence_scope']=dict(applied=True,method='learned QA answerability + relevance + embeddings; no question-type regex')
    result['generic_candidate_checks']=[];result['rag_attempts']=[]
    # Extract candidate answers from evidence, not from the original output.
    extracted=[]
    result['qa_extraction_errors']=[]
    for row in evidence[:int(cfg['max_evidence'])]:
        try:qa=models.qa(question,row['passage'])
        except Exception as exc:
            result['qa_extraction_errors'].append(dict(source_id=row['id'],error=str(exc)));continue
        if qa['answer'] and qa['answerability']>=cfg['qa_answerability'] and qa['null_margin']>=cfg['qa_margin']:
            extracted.append(dict(row=row,qa=qa))
    # Preserve existing TAG generation for presentation; independently extracted answers
    # provide a fallback if TAG changes a correct relation or abstains unnecessarily.
    candidates=[]
    context=sorted(extracted,key=lambda x:x['qa']['answerability'],reverse=True)
    selected=[x['row'] for x in context[:3]]
    if selected:
        prompt=api.rag_prompt(question,selected,0)
        try:
            raw,_=tag.generate(prompt,140);candidate,error=api.clean_candidate(raw)
            result['rag_attempts'].append(dict(raw_output=raw,candidate=candidate or None,error=error))
            if candidate:candidates.append(dict(text=candidate,origin='TAG_evidence_generation',prompt=prompt))
        except Exception as exc:result['rag_attempts'].append(dict(error=str(exc)))
    # Extractive fallback: verify a complete source sentence, not a bare answer fragment.
    # The extracted short answer is reported separately for inspection.
    for item in context:
        row=item['row'];qa=item['qa'];text=row['passage'];start=qa['start'];end=qa['end']
        left=max(text.rfind('.',0,start),text.rfind('!',0,start),text.rfind('?',0,start))+1
        stops=[x for x in [text.find('.',end),text.find('!',end),text.find('?',end)] if x>=0]
        right=min(stops)+1 if stops else len(text)
        sentence=text[left:right].strip()
        if not sentence or len(sentence.split())>cfg['max_answer_words']:continue
        candidates.append(dict(text=sentence,short_answer=qa['answer'],origin='extractive_QA_source_sentence',
            source_id=row['id'],prompt=api.rag_prompt(question,[row],0)))
    original=result.get('original_tag_output','')
    if original and not api.unknown(original):
        candidates.append(dict(text=original,origin='original_candidate_evidence_verification',
            prompt=api.rag_prompt(question,selected or evidence[:2],0)))
    seen=set();accepted=[]
    for candidate in candidates:
        text=candidate['text']
        if text in seen:continue
        seen.add(text)
        try:report=verify(question,text,evidence,pairs,args)
        except Exception as exc:
            result['generic_candidate_checks'].append(dict(candidate=text,origin=candidate['origin'],
                verification=dict(accepted=False,overall_score=0.,support_ids=[],error=f'{type(exc).__name__}: {exc}')))
            continue
        check=dict(candidate=text,origin=candidate['origin'],verification=report)
        if report['accepted'] and original and not api.unknown(original):
            try:
                comp=comparison(original,text,pairs,models,cfg);check['original_comparison']=comp
                if comp['contradiction_detected']:
                    # Stronger independent second-NLI support required for correction.
                    supporters=[x for x in report['passage_checks'] if x.get('supports')]
                    additional=any(x['nli2']['entailment']>=cfg['extra_original_conflict'] and
                        x['answer_agreement']['equivalent'] for x in supporters)
                    check['additional_correction_check']=dict(passed=additional,
                        rule='At least one supporting passage passes independent NLI2 entailment floor plus QA answer agreement; original is NOT ground truth.')
                    report['gates']['additional_correction_check']=additional
                    if not additional:
                        report['accepted']=False;report['reason']='additional_original_conflict_check_failed'
            except Exception as exc:
                report['accepted']=False;report['gates']['additional_correction_check']=False
                report['reason']='additional_original_conflict_check_error';check['comparison_error']=str(exc)
        result['generic_candidate_checks'].append(check)
        if report['accepted']:accepted.append((report['overall_score'],candidate,report,check))
    if not accepted:
        result['final_answer']='abstain';result['action']='abstain';result['sources']=[]
        result['verification']=dict(accepted=False,reason='No candidate passed generic QA + dual NLI + conflict checks',
            candidate_count=len(result['generic_candidate_checks']),settings=cfg)
        if result['generic_candidate_checks']:
            best=max(result['generic_candidate_checks'],key=lambda x:x['verification']['overall_score'])
            result['rag_candidate']=best['candidate'];result['verification']=best['verification']
            prompt=next((x['prompt'] for x in candidates if x['text']==best['candidate']),api.rag_prompt(question,evidence[:2],0))
            try:result['rag_probe_diagnostic']=tag.measure(prompt,best['candidate'],rag=True)
            except Exception as exc:result['rag_probe_diagnostic']=dict(available=False,error=str(exc),excluded_from_rag_release=True)
        return result
    _,candidate,report,check=max(accepted,key=lambda x:x[0])
    result.update(final_answer=candidate['text'],action='answer_evidence_verified',rag_candidate=candidate['text'],
        final_answer_origin=candidate['origin'],verification=report,
        sources=[x for x in evidence if x['id'] in report['support_ids']],
        original_comparison=check.get('original_comparison'),
        additional_correction_check=check.get('additional_correction_check'))
    if candidate.get('short_answer'):result['extracted_short_answer']=candidate['short_answer']
    try:result['rag_probe_diagnostic']=tag.measure(candidate['prompt'],candidate['text'],rag=True)
    except Exception as exc:result['rag_probe_diagnostic']=dict(available=False,error=str(exc),excluded_from_rag_release=True)
    return result


def reevaluate_file(a):
    """Real fresh NLP inference on saved passages; no TAG/GPU required."""
    import tag_rag_v3 as api
    data=json.loads(Path(a.input).read_text());args=argparse.Namespace(**data['thresholds'])
    args.final_threshold=data['thresholds']['final']
    pairs=api.PairModels();question=data['question'];answer=data.get('rag_candidate')
    if not answer:raise ValueError('Saved report has no rag_candidate')
    report=verify(question,answer,data['retrieved_passages'],pairs,args)
    original=data.get('original_tag_output','')
    if report['accepted'] and original and not api.unknown(original):
        comp=comparison(original,answer,pairs,extra_models(),settings())
        report['original_comparison']=comp
        if comp['contradiction_detected']:
            additional=any(x.get('supports') and x['nli2']['entailment']>=settings()['extra_original_conflict'] for x in report['passage_checks'])
            report['gates']['additional_correction_check']=additional
            report['accepted']=report['accepted'] and additional
    result=dict(mode='fresh_NLP_verification_of_saved_candidate_not_fresh_retrieval',question=question,
        candidate=answer,verification=report,original_final_answer=data.get('final_answer'))
    text=json.dumps(result,indent=2)
    if Path(a.input).resolve()==Path(a.output).resolve():raise ValueError('Do not overwrite input')
    Path(a.output).write_text(text);print(text)

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run generic QA/NLI verification on saved report (downloads verifier models, no TAG GPU required).')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    reevaluate_file(p.parse_args())
