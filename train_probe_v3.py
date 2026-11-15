""" CuratorKIT -> TAG activation extraction -> probe -> AuditKIT.

"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score,brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tag_rag_v3 import TAG,BASE_PROMPT,MODEL,norm


def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def versions():
    return {name:importlib.metadata.version(name) for name in ['torch','transformers','scikit-learn','curatorkit','auditkit']}
def read(path):
    d=pd.read_csv(path,dtype={'question_id':str},keep_default_na=False)
    required=['question_id','split','question','model_answer','correctness_label']
    if not set(required)<=set(d):raise ValueError('Missing required columns')
    for c in required[:-1]:
        if d[c].astype(str).str.strip().eq('').any():raise ValueError(f'Empty {c}')
    y=pd.to_numeric(d.correctness_label,errors='raise')
    if not y.isin([-1,0,1]).all():raise ValueError('Labels must be -1/0/1')
    d['correctness_label']=y.astype(int);d['key']=d.question.map(lambda s:norm(s).casefold())
    if d.question_id.duplicated().any() or d.key.duplicated().any():raise ValueError('Duplicate IDs/questions; resolve grouped splits first')
    if set(d.split)!={'train','validation','test'}:raise ValueError('Expected existing train/validation/test')
    return d

def prepare(a):
    from curatorkit import Curator,CuratorConfig
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'prepared.csv').exists() or (out/'curator').exists():raise ValueError('Use a fresh output directory')
    d=read(a.csv);src=out/'curator_input.jsonl'
    with src.open('w') as f:
        for q in d.loc[d.split=='train','question']:
            f.write(json.dumps(dict(instruction=q,input='',output='activation probe question'))+'\n')
    dest=out/'curator'
    result=Curator(CuratorConfig(dataset=str(src),dedup='exact',clean=False,
        export_formats=['alpaca'],output_dir=str(dest))).run()
    result.print_summary();keys=set()
    for p in dest.rglob('*'):
        if not p.is_file() or p.suffix not in ['.json','.jsonl'] or 'alpaca' not in p.name.lower():continue
        data=json.loads(p.read_text()) if p.suffix=='.json' else [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if isinstance(data,dict):data=data.get('data',[])
        for row in data:
            if isinstance(row,dict) and isinstance(row.get('instruction'),str):keys.add(norm(row['instruction']).casefold())
    if not keys or not keys<=set(d.loc[d.split=='train','key']):raise ValueError('Unexpected Curator export; no silent bypass')
    before=len(d);d=d[(d.split!='train')|d.key.isin(keys)].reset_index(drop=True)
    d.to_csv(out/'prepared.csv',index=False)
    d[d.correctness_label==-1].to_csv(out/'unknown_labels_for_review.csv',index=False)
    save(out/'data_report.json',dict(source_rows=before,retained_rows=len(d),source_sha256=digest(a.csv),
        counts=pd.crosstab(d.split,d.correctness_label).to_dict()))

def extract(a):
    import torch
    from transformers import AutoConfig
    out=Path(a.out);d=read(out/'prepared.csv')
    cfg=AutoConfig.from_pretrained(MODEL,revision=a.revision);n=cfg.num_hidden_layers
    dtype=a.dtype if a.dtype!='auto' else ('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ('float16' if torch.cuda.is_available() else 'float32'))
    meta=dict(model=MODEL,revision=getattr(cfg,'_commit_hash',None) or a.revision,dtype=dtype,
        layers=sorted(set([n//3,2*n//3,n-1])),max_tokens=a.max_tokens,prompt=BASE_PROMPT,
        prepared_sha256=digest(out/'prepared.csv'),dependencies=versions())
    tag=TAG(dict(meta=meta),a.device);meta['max_tokens']=tag.limit
    path=out/'feature_meta.json'
    if path.exists() and json.loads(path.read_text())!=meta:raise ValueError('Changed extraction config/data/environment; use new output directory')
    save(path,meta);folder=out/'activation_rows';folder.mkdir(exist_ok=True)
    for i,row in d.iterrows():
        dest=folder/f'{i:06d}.npy'
        if dest.exists():continue
        prompt=BASE_PROMPT.replace('{question}',row.question);batch=tag.inputs(prompt);n=batch['input_ids'].shape[1]
        ans=tag.tok.encode(row.model_answer,add_special_tokens=False)
        if not ans or n+len(ans)>tag.limit:raise ValueError(f'Empty/oversized replay at row {i}')
        ids=torch.cat([batch['input_ids'],torch.tensor([ans],device=tag.device)],dim=1)
        cache={};handles=[]
        def hook(layer):
            def capture(module,args,out):
                h=out[0] if isinstance(out,tuple) else out
                cache[layer]=torch.stack([h[0,n-1],h[0,n:].mean(0),h[0,-1]]).detach().float().cpu().numpy()
            return capture
        try:
            for layer in tag.layers:handles.append(tag.blocks[layer].register_forward_hook(hook(layer)))
            with torch.inference_mode():tag.backbone(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        finally:
            for h in handles:h.remove()
        temp=folder/f'{i:06d}.tmp.npy';np.save(temp,np.stack([cache[l] for l in tag.layers]));temp.replace(dest)
        if (i+1)%25==0:print(f'Extracted {i+1}/{len(d)}',flush=True)
    x=np.stack([np.load(folder/f'{i:06d}.npy',allow_pickle=False) for i in range(len(d))])
    np.savez_compressed(out/'features.npz',x=x,ids=d.question_id.to_numpy(dtype=str))

def matrix(x,l,m):return x[:,l,0,:] if m=='prompt' else (x[:,l,1,:] if m=='answer' else x[:,l].reshape(len(x),-1))
def classifier(c,seed):return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=seed))
def outcome(y,keep):
    y=np.asarray(y);keep=np.asarray(keep,bool);substantive=keep&(y!=-1);n=len(y);answered=int(substantive.sum())
    wrong=int(((y==0)&substantive).sum());correct=int(((y==1)&substantive).sum())
    return dict(n=n,coverage=answered/n,answered=answered,wrong_delivered=wrong,correct_delivered=correct,
        wrong_per_question=wrong/n,selective_risk=wrong/answered if answered else None,
        released_unknown=int(((y==-1)&keep).sum()))

def train(a):
    import auditkit as ak
    out=Path(a.out);(out/'TRAINING_COMPLETE.json').unlink(missing_ok=True)
    d=read(out/'prepared.csv');meta=json.loads((out/'feature_meta.json').read_text())
    if digest(out/'prepared.csv')!=meta['prepared_sha256']:raise ValueError('Prepared data changed')
    z=np.load(out/'features.npz',allow_pickle=False);x=z['x'];y=d.correctness_label.to_numpy();binary=(y==1).astype(int)
    if not np.array_equal(z['ids'],d.question_id.to_numpy(dtype=str)):raise ValueError('Row mismatch')
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    tr=np.flatnonzero(d.split=='train');va=np.flatnonzero(d.split=='validation');te=np.flatnonzero(d.split=='test')
    fit,tune=train_test_split(tr,test_size=.25,stratify=binary[tr],random_state=a.seed)
    cal,policy=train_test_split(va,test_size=.5,stratify=binary[va],random_state=a.seed)
    candidates=[]
    for l in range(x.shape[1]):
        for m in ['prompt','answer','joint']:
            xx=matrix(x,l,m)
            for c in [.001,.01,.1,1.]:
                p=classifier(c,a.seed).fit(xx[fit],binary[fit])
                loss=log_loss(binary[tune],p.predict_proba(xx[tune])[:,1],labels=[0,1])
                candidates.append(dict(layer_index=l,mode=m,C=c,loss=float(loss)))
    selected=min(candidates,key=lambda r:r['loss']);xx=matrix(x,selected['layer_index'],selected['mode'])
    probe=classifier(selected['C'],a.seed).fit(xx[tr],binary[tr]);raw=probe.decision_function(xx).reshape(-1,1)
    calibrator=LogisticRegression(C=1.,max_iter=1000).fit(raw[cal],binary[cal]);p=calibrator.predict_proba(raw)[:,1]
    from tag_rag_v3 import unknown
    flags=d.model_answer.map(unknown).to_numpy(bool);curve=[]
    for t in np.unique(np.r_[0.,p[policy],1.000001]):
        curve.append(dict(outcome(y[policy],(p[policy]>=t)&~flags[policy]),threshold=float(t)))
    feasible=[r for r in curve if r['coverage']>=a.min_coverage and r['selective_risk'] is not None and r['selective_risk']<=a.max_risk]
    best=dict(max(feasible,key=lambda r:r['coverage']),constraints_met=True) if feasible else dict(threshold=1.000001,constraints_met=False)
    pd.DataFrame(curve).to_csv(out/'validation_risk_coverage.csv',index=False)
    keep=(p[te]>=best['threshold'])&~flags[te]
    report=dict(baseline=outcome(y[te],np.ones(len(te),bool)),gated=outcome(y[te],keep),
        brier=float(brier_score_loss(binary[te],p[te])),auroc=float(roc_auc_score(binary[te],p[te])),
        validation_policy=best,selected=selected,
        split_ids={k:d.iloc[v].question_id.tolist() for k,v in [('fit',fit),('tune',tune),('calibration',cal),('policy',policy),('test',te)]})
    save(out/'test_report.json',report)
    predictions=d.iloc[te][['question_id','correctness_label']].copy();predictions['p_correct']=p[te];predictions['release']=keep
    predictions.to_csv(out/'test_predictions.csv',index=False)
    lookup=dict(zip(d.iloc[te].question_id,keep.astype(int).astype(str)))
    samples=[ak.Sample(input=str(row.question_id),target=str(int(row.correctness_label==1))) for row in d.iloc[te].itertuples()]
    audit=ak.evaluate(samples,model=lambda prompts:[lookup[str(q)] for q in prompts],scorers='exact_match')
    if getattr(audit,'errors',None):raise RuntimeError(str(audit.errors))
    save(out/'auditkit_headline.json',audit.headline);(out/'auditkit_summary.txt').write_text(str(audit.summary()))
    joblib.dump(dict(probe=probe,calibrator=calibrator,selected=selected,threshold=best['threshold'],
        policy=best,meta=meta,dependencies=versions()),out/'probe.joblib')
    save(out/'TRAINING_COMPLETE.json',dict(auditkit_completed=True,constraints_met=best['constraints_met']))
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','extract','train']:
        s=sub.add_parser(name);s.add_argument('--out',default='artifacts_v3')
        if name=='prepare':s.add_argument('--csv',required=True)
        if name=='extract':
            s.add_argument('--revision',default='main');s.add_argument('--device',default='cuda')
            s.add_argument('--dtype',choices=['auto','bfloat16','float16','float32'],default='auto');s.add_argument('--max-tokens',type=int,default=8192)
        if name=='train':
            s.add_argument('--seed',type=int,default=42);s.add_argument('--min-coverage',type=float,required=True);s.add_argument('--max-risk',type=float,required=True)
    a=p.parse_args()
    if a.command=='train' and not(0<a.min_coverage<=1 and 0<=a.max_risk<=1):p.error('Invalid policy constraints')
    globals()[a.command](a)
if __name__=='__main__':main()
