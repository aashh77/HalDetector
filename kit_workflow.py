"""Two mandatory KITs in the live retrieval/release path.
Reuse the working QA/embedding/NLI/probe models; no model replacement or retraining.
"""
import copy
import math
import time
from kit_adapters import require_kits,curate_evidence,audited_source,release_audit,KitFailure

def finalize(result,question,evidence,tag,pairs,args,api):
    import generic_verifier as g
    import simple_source_workflow as base
    packages=require_kits()
    started=time.perf_counter()
    curated,curation=curate_evidence(evidence) # actual CuratorKIT pipeline
    models=g.extra_models();cfg=g.settings()
    if hasattr(api,'DEFINITIONS'):
        api.DEFINITIONS['final_overall']='Mandatory CuratorKIT exact evidence curation + AuditKIT custom source metrics/release contract. Best-source policy; no cross-source consensus. Different answer: evidence score. Matching: .8*evidence+.2*(w*probe+(1-w)*L). Not calibrated truth.'
    result['workflow_version']='two_kits_required_v1'
    result['kit_execution']=dict(packages=packages,curatorkit=curation,
        auditkit=dict(status='pending'),sdk_failure_policy='fail_closed_no_fallback')
    result['source_selection_policy']=dict(consensus_required=False,cross_source_contradiction_veto=False,
        warning='Best-source policy ignores unselected disagreement; KITs do not establish source truth.')
    records=[]
    for row in curated:
        record=base.score_source(question,row,models,pairs,args,cfg,g.provenance)
        records.append(audited_source(question,record)) # actual SDK-returned metrics consumed
    # Keep duplicate IDs and metadata visible, without pretending another model call occurred.
    byid={r['source_id']:r for r in records}
    for duplicate in curation.get('duplicate_sources',[]):
        original=next(r for r in evidence if r['id']==duplicate['source_id'])
        representative=byid[duplicate['duplicate_of']]
        records.append(dict(source_id=original['id'],title=original.get('title'),url=original.get('url'),
            passage=original['passage'],answer=representative.get('answer'),eligible=False,score=None,
            curation_status='exact_duplicate_not_rescored',duplicate_of=duplicate['duplicate_of'],
            model_outputs=None,auditkit=dict(status='not_scored_exact_duplicate')))
    order={r['id']:i for i,r in enumerate(evidence)};records.sort(key=lambda r:order[r['source_id']])
    result['per_source_results']=records
    selected,reason=base.choose(records,args.final_threshold)
    result['selected_rag']=dict(answer=selected['answer'] if selected else None,
        source_ids=[selected['source_id']] if selected else [],evidence_score=selected['score'] if selected else None,
        threshold=args.final_threshold,selection_reason=reason,consensus_required=False,
        cross_source_contradiction_veto=False,score_executor='auditkit_custom_scorers')
    result.update(rag_candidate=selected['answer'] if selected else None,rag_probe_diagnostic=None,
        rag_answer_match=None,rag_final_scoring=dict(branch='no_accepted_source',final_score=None,accepted=False),
        final_answer='abstain',action='abstain',sources=[],final_score=None)
    result['verification']=dict(accepted=False,reason=reason,overall_score=selected['score'] if selected else None,
        support_ids=[],per_source_results=records,kit_execution=result['kit_execution'])
    if reason=='selected_source_passed':
        answer=selected['answer'];source=next(r for r in curated if r['id']==selected['source_id'])
        prompt=api.rag_prompt(question,[source],0)
        try:probe=tag.measure(prompt,answer,rag=True)
        except Exception as exc:probe=dict(available=False,error=str(exc),excluded_from_rag_release=True)
        result['rag_probe_diagnostic']=probe
        try:match=base.answer_match(question,result.get('original_tag_output',''),answer,models,cfg,g.agreement)
        except Exception as exc:match=dict(matches=False,status='match_unavailable',error=str(exc))
        result['rag_answer_match']=match;S=selected['score'];w=float(getattr(args,'probe_weight',.8))
        if not 0<=w<=1:raise KitFailure('Invalid probe weight')
        if match['matches'] and probe.get('available'):
            P=float(probe['p_correct']);L=float(probe['token_likelihood'])
            if not all(math.isfinite(v) and 0<=v<=1 for v in [P,L]):raise KitFailure('Invalid probe/L values')
            PL=w*P+(1-w)*L;F=.8*S+.2*PL
            scoring=dict(branch='matching_answer_evidence_plus_probe_likelihood',probe_score=P,likelihood_score=L,
                probe_likelihood_score=PL,evidence_score=S,final_score=F,threshold=args.final_threshold,
                accepted=F>=args.final_threshold,formula='F=.8*S+.2*(w*P_RAG+(1-w)*L_RAG)')
        else:
            F=S
            scoring=dict(branch='matching_answer_probe_unavailable_evidence_only' if match['matches'] else 'different_answer_selected_source_evidence_only',
                evidence_score=S,final_score=F,threshold=args.final_threshold,accepted=True,formula='F=S')
        scoring['score_is_calibrated_probability']=False
        result['rag_final_scoring']=scoring;result['final_score']=F
        result['verification'].update(accepted=scoring['accepted'],reason='final_score_passed' if scoring['accepted'] else 'matching_branch_below_threshold',
            overall_score=F,support_ids=[source['id']] if scoring['accepted'] else [],final_scoring=scoring)
        if scoring['accepted']:result.update(final_answer=answer,action='answer_selected_source',sources=[source],
            final_answer_origin='exact_QA_span_curated_by_CuratorKIT_scored_by_AuditKIT')
    result['kit_execution']['auditkit']=dict(status='executed',source_audits=sum(r.get('auditkit',{}).get('status')=='executed' for r in records))
    result['kit_execution']['retrieval_stage_seconds']=time.perf_counter()-started
    return result

def guarded_run(original_run,args):
    """AuditKIT release check runs for BOTH retrieval and original-only branches."""
    packages=require_kits() # no silent disabled-KIT mode
    result=original_run(args)
    if not isinstance(result,dict):raise KitFailure('Pipeline returned a non-object')
    result.setdefault('kit_execution',dict(packages=packages,
        curatorkit=dict(status='not_completed_pipeline_error' if result.get('retrieval_done') else 'not_applicable_no_retrieval'),auditkit=dict(status='pending'),
        sdk_failure_policy='fail_closed_no_fallback'))
    if result.get('error'):
        result['kit_execution']['auditkit']['release_status']='not_executed_pipeline_error'
        result.update(final_answer='abstain',sources=[],action='abstain',final_score=None)
        return result
    try:
        report=release_audit(result,args)
    except Exception as exc:
        result['kit_execution']['auditkit']['release_status']='failed'
        result['kit_execution']['auditkit']['error']=f'{type(exc).__name__}: {exc}'
        result.update(final_answer='abstain',sources=[],action='abstain',final_score=None,
            error='Mandatory KIT release audit failed; inspect kit_execution. No unaudited answer released.')
        if not isinstance(result.get('verification'),dict):result['verification']={}
        result['verification'].update(accepted=False,reason='mandatory_kit_release_failed',support_ids=[])
    else:
        result['kit_execution']['auditkit'].update(status='executed',release_status='passed',release=report)
    if not isinstance(result.get('verification'),dict):result['verification']={}
    result['verification']['kit_execution']=result['kit_execution']
    return result

def compact_result(full):
    from simple_source_workflow import compact_result as base_compact
    compact=base_compact(full)
    # Preserve original sequence, expose mandatory KIT reports without hiding details.
    compact['kit_execution']=copy.deepcopy(full.get('kit_execution'))
    return compact
