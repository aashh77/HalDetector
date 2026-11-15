"""Freeze official Hub Parquet test questions/answers, never silently use validation."""
import argparse,hashlib,json
from pathlib import Path

def fetch(a):
    from huggingface_hub import HfApi,hf_hub_download
    from datasets import load_dataset
    api=HfApi();info=api.dataset_info(a.repo,revision=a.revision);sha=info.sha
    files=api.list_repo_files(a.repo,repo_type='dataset',revision=sha)
    prefix=f'{a.config}/{a.split}/'
    shards=sorted(x for x in files if x.startswith(prefix) and x.endswith('.parquet'))
    if not shards:raise RuntimeError(f'No parquet shards under {prefix} at {sha}. Choose an explicit config/split; no automatic fallback.')
    local=[hf_hub_download(a.repo,f,repo_type='dataset',revision=sha) for f in shards]
    ds=load_dataset('parquet',data_files=local,split='train',streaming=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    tmp=out.with_suffix('.partial');count=0;ids=set();missing=0
    with tmp.open('w') as f:
        for row in ds:
            ans=row.get('answer') or {};value=ans.get('value','')
            aliases=list(dict.fromkeys([value]+ans.get('aliases',[])+ans.get('normalized_aliases',[])))
            aliases=[str(x) for x in aliases if str(x).strip()]
            if not value or not aliases:missing+=1;continue
            qid=str(row['question_id'])
            if qid in ids:continue
            ids.add(qid)
            f.write(json.dumps(dict(id=qid,question=row['question'],GT=value,aliases=aliases),ensure_ascii=False)+'\n');count+=1
            if a.limit and count>=a.limit:break
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'{missing} rows lacked labeled answers. No test metrics can be computed on hidden labels. Use an explicitly named labeled split; do not call it test.')
    if not count:tmp.unlink(missing_ok=True);raise RuntimeError('No labeled questions')
    tmp.replace(out)
    manifest=dict(repo=a.repo,config=a.config,split=a.split,revision=sha,shards=shards,n=count,
        subset_limit=a.limit,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        note='Only questions/canonical answers/aliases exported. No dataset evidence enters the inference path.')
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='mandarjoshi/trivia_qa')
    p.add_argument('--config',default='rc.nocontext');p.add_argument('--split',default='validation')
    p.add_argument('--revision',default='refs/convert/parquet');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--output',default='data/triviaqa_val.jsonl');a=p.parse_args()
    if a.limit<0:p.error('limit must be nonnegative')
    fetch(a)
