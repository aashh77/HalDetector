"""Mandatory real SDK calls. No fallback reimplementation of either toolkit.
CuratorKIT curates byte-identical passages; AuditKIT executes custom policy metrics.
These SDKs are not additional factuality models and do not establish source truth.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import tempfile
import time

class KitFailure(RuntimeError):pass

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def package_info(name):
    dist=importlib.metadata.distribution(name)
    direct=dist.read_text('direct_url.json')
    return dict(version=dist.version,direct_url=json.loads(direct) if direct else None)

@lru_cache(maxsize=1)
def require_kits():
    # Import BOTH at startup, not only after a result was delivered.
    from curatorkit import Curator,CuratorConfig
    import auditkit as ak
    for name in ['Sample','evaluate','scorer','Direction']:
        if not hasattr(ak,name):raise KitFailure('Incompatible AuditKIT API: '+name)
    info={name:package_info(name) for name in ['curatorkit','auditkit']}
    lock=Path(__file__).with_name('kits.lock.json')
    if not lock.exists():raise KitFailure('Run python kit_preflight.py --write-lock before installing/enabling the hook')
    if json.loads(lock.read_text())['packages']!=info:raise KitFailure('KIT environment changed after contract validation; rerun preflight deliberately')
    return info

def curate_evidence(evidence):
    """Use CuratorKIT's public pipeline. No LLM, no text rewrite, no near dedup.
    All byte-distinct passages must survive; an unexpected deletion fails closed.
    """
    from curatorkit import Curator,CuratorConfig
    start=time.perf_counter()
    ids=[r['id'] for r in evidence]
    if len(ids)!=len(set(ids)):raise KitFailure('Duplicate source IDs')
    if any(not isinstance(r.get('passage'),str) or not r['passage'].strip() for r in evidence):raise KitFailure('Empty/nontext evidence passage')
    if not evidence:return [],dict(status='not_applicable_empty_evidence',input_count=0,output_count=0)
    # Stable representative for duplicates: highest existing relevance, then input order.
    ordered=sorted(enumerate(evidence),key=lambda x:(-float(x[1].get('relevance',0.)),x[0]))
    payload=[dict(instruction=r['passage'],input='',output=r['passage']) for _,r in ordered]
    with tempfile.TemporaryDirectory(prefix='kit_curate_') as folder:
        root=Path(folder);source=root/'passages.jsonl'
        with source.open('w',encoding='utf-8') as f:
            for row in payload:f.write(json.dumps(row,ensure_ascii=True)+'\n')
        config=CuratorConfig(dataset=str(source),dedup='exact',clean=False,
            export_formats=['alpaca'],output_dir=str(root/'curated'))
        result=Curator(config).run() # REQUIRED SDK execution, not an import-only claim.
        passed=result.passed
        if not isinstance(passed,(list,tuple)):raise KitFailure('CuratorKIT result.passed must be sample objects; unsupported SDK contract')
        texts=[]
        for sample in passed:
            text=sample.instruction
            if sample.output!=text:raise KitFailure('CuratorKIT changed the identity payload; source text must remain exact')
            texts.append(text)
        expected={r['passage'] for r in evidence}
        if len(texts)!=len(set(texts)) or set(texts)!=expected:
            raise KitFailure('CuratorKIT did not perform exact-only deduplication. Distinct/qualified evidence may have been removed; refusing release.')
        # Persist hashes of actual SDK provenance/export files in the result, not temp paths.
        artifacts={str(p.relative_to(root/'curated')):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'curated').rglob('*') if p.is_file()}
        if not artifacts:raise KitFailure('CuratorKIT produced no provenance/export files')
        provenance_documents={}
        for name in ['manifest.json','lexsi_provenance.json']:
            path=root/'curated'/name
            if path.exists():
                try:provenance_documents[name]=json.loads(path.read_text())
                except (ValueError,UnicodeError):raise KitFailure('Invalid SDK provenance JSON: '+name)
        stage_counts=getattr(result,'stage_counts',None)
        stage_counts=json.loads(json.dumps(stage_counts,default=str))
    first={}
    for _,row in ordered:first.setdefault(row['passage'],row)
    kept=[first[text] for text in texts]
    kept.sort(key=lambda r:ids.index(r['id']))
    kept_ids={r['id'] for r in kept}
    duplicates=[dict(source_id=r['id'],duplicate_of=first[r['passage']]['id']) for r in evidence if r['id'] not in kept_ids]
    return kept,dict(status='executed',sdk='curatorkit',operation='Curator(CuratorConfig(dedup=exact, clean=False)).run',
        input_count=len(evidence),output_count=len(kept),kept_ids=[r['id'] for r in kept],
        duplicate_sources=duplicates,stage_counts=stage_counts,artifact_sha256=artifacts,provenance_documents=provenance_documents,
        input_sha256=digest(evidence),output_sha256=digest(kept),seconds=time.perf_counter()-start,
        note='Only identical passage copies are removed. Distinct contradictory passages are retained; no source truth or independence is inferred.')

def audit_values(question,payload,values,calculators=None):
    """Real AuditKIT custom scorers, with finite [0,1] output contracts.
    Metric values originate from existing NLP models/policy checks, not KIT knowledge.
    The SDK-returned headline values are consumed by ranking/release.
    """
    import auditkit as ak
    start=time.perf_counter()
    for name,value in values.items():
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise KitFailure('Invalid policy metric: '+name)
    body=dict(payload=payload,metrics=values)
    serialized=json.dumps(body,sort_keys=True,ensure_ascii=True,allow_nan=False)
    fingerprint=digest(dict(question=question,body=body))
    scorers=[]
    for name in values:
        def metric(sample,output,_name=name):
            parsed=json.loads(str(output))
            return float(calculators[_name](parsed['payload']) if calculators else parsed['metrics'][_name])
        metric.__name__=name
        scorers.append(ak.scorer(direction=ak.Direction.MAXIMIZE)(metric))
    samples=[ak.Sample(input=json.dumps(dict(question=question,trace_sha256=fingerprint)),target='')]
    run=ak.evaluate(samples,model=lambda prompts:[serialized for _ in prompts],scorers=scorers)
    if getattr(run,'errors',None):raise KitFailure('AuditKIT reported errors: '+str(run.errors))
    headline=getattr(run,'headline',None)
    if not isinstance(headline,dict):raise KitFailure('AuditKIT headline must be a metric-to-number mapping')
    actual={}
    for name,expected in values.items():
        value=headline.get(name)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value-expected)>1e-8:
            raise KitFailure('AuditKIT scorer contract mismatch for '+name+': '+repr(value))
        actual[name]=float(value)
    return actual,dict(status='executed',sdk='auditkit',headline=actual,
        fingerprint=str(getattr(run,'fingerprint','')),trace_sha256=fingerprint,
        seconds=time.perf_counter()-start,
        note='Custom scorers execute deterministic checks over model outputs; not an independent truth judge or extra NLP model.')

def audited_source(question,record):
    if record.get('error'):
        # Model errors are not turned into a successful audit or an eligible source.
        record['auditkit']=dict(status='not_scored_model_error');record['eligible']=False;return record
    m=record['metrics'];clip=lambda x:max(0.,min(1.,float(x)))
    score=.35*float(m['qa_answerability'])+.25*clip(float(m['qa_margin'])/15.)+.20*clip(m['question_passage_cosine'])+.10*clip(m['query_passage_relevance'])+.10*clip(m['source_sentence_entailment'])
    values=dict(qa_answerability=clip(m['qa_answerability']),qa_margin_index=clip(float(m['qa_margin'])/15.),
        semantic_linkage=clip(m['question_passage_cosine']),query_relevance=clip(m['query_passage_relevance']),
        nli_fidelity=clip(m['source_sentence_entailment']),exact_span=float(bool(m['exact_span'])),
        source_eligible=float(all(record['gates'].values())),evidence_index=score)
    def index(p):
        x=p['metrics']
        return .35*float(x['qa_answerability'])+.25*clip(float(x['qa_margin'])/15.)+.20*clip(x['question_passage_cosine'])+.10*clip(x['query_passage_relevance'])+.10*clip(x['source_sentence_entailment'])
    def exact(p):
        qa=p['model_outputs']['deepset/roberta-base-squad2']
        start=qa['start'];end=qa['end']
        return float(isinstance(start,int) and isinstance(end,int) and 0<=start<end<=len(p['passage']) and p['passage'][start:end]==p['answer'])
    calculators=dict(
        qa_answerability=lambda p:clip(p['metrics']['qa_answerability']),
        qa_margin_index=lambda p:clip(float(p['metrics']['qa_margin'])/15.),
        semantic_linkage=lambda p:clip(p['metrics']['question_passage_cosine']),
        query_relevance=lambda p:clip(p['metrics']['query_passage_relevance']),
        nli_fidelity=lambda p:clip(p['metrics']['source_sentence_entailment']),
        exact_span=exact,source_eligible=lambda p:float(all(p['gates'].values()) and exact(p)==1),evidence_index=index)
    # Calculations execute INSIDE AuditKIT scorers over serialized raw model output.
    # 'values' are independent expected results for detecting SDK/API corruption.
    actual,report=audit_values(question,record,values,calculators=calculators)
    record['score']=actual['evidence_index'] # ranking consumes actual SDK result
    record['eligible']=actual['source_eligible']==1 and actual['exact_span']==1
    record['auditkit']=report
    return record

def release_contract(result,args):
    """Pure contract predicate, executed inside AuditKIT at release."""
    accepted=result.get('final_answer','abstain') not in ('','abstain')
    retrieval=bool(result.get('retrieval_done'))
    if retrieval:
        selected=result.get('selected_rag') or {};ids=selected.get('source_ids',[])
        records=result.get('per_source_results',[])
        matching=[r for r in records if r['source_id'] in ids]
        source=next((s for s in result.get('sources',[]) if s['id'] in ids),None)
        row=matching[0] if len(matching)==1 else None
        span=bool(row and source and result['final_answer']==row.get('answer') and result['final_answer'] in source['passage'])
        score=result.get('final_score');threshold=args.final_threshold
        contract=(not accepted) or bool(span and row['eligible'] and row.get('auditkit',{}).get('status')=='executed'
            and result.get('rag_final_scoring',{}).get('accepted') and isinstance(score,(int,float)) and math.isfinite(score) and score>=threshold)
    else:
        metrics=result.get('original_metrics',{});score=metrics.get('overall_score')
        contract=(not accepted) or bool(result['final_answer']==result.get('original_tag_output') and
            isinstance(score,(int,float)) and math.isfinite(score) and score>=args.original_threshold)
    return bool(contract)

def release_audit(result,args):
    accepted=result.get('final_answer','abstain') not in ('','abstain')
    contract=release_contract(result,args)
    values=dict(release_contract=float(contract),answer_released=float(accepted))
    actual,report=audit_values(result.get('question',''),result,values,calculators=dict(
        release_contract=lambda p:float(release_contract(p,args)),
        answer_released=lambda p:float(p.get('final_answer','abstain') not in ('','abstain'))))
    if actual['release_contract']!=1:raise KitFailure('AuditKIT final release contract failed')
    return report
