"""Paired TriviaQA benchmark. Results can improve, tie, or regress; never fabricated."""
import argparse,csv,hashlib,json,os,platform,statistics,subprocess,threading,time
from pathlib import Path
from metrics import score,summarize,normalize

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bytes_on_disk(root):
    root=Path(root)
    if not root.exists():return 0
    # Count each physical inode once (Hub snapshot symlinks share blobs).
    seen=set();total=0
    for p in root.rglob('*'):
        try:
            st=p.stat();key=(st.st_dev,st.st_ino)
            if p.is_file() and key not in seen:seen.add(key);total+=st.st_size
        except OSError:pass
    return total
class Monitor:
    def __init__(self):self.stop=threading.Event();self.rss=0;self.gpu=0;self.gpu_observed=False
    def __enter__(self):
        import psutil
        self.proc=psutil.Process();self.cpu0=sum(self.proc.cpu_times()[:2]);self.t0=time.perf_counter()
        def poll():
            while not self.stop.is_set():
                try:
                    processes=[self.proc]+self.proc.children(recursive=True)
                    self.rss=max(self.rss,sum(p.memory_info().rss for p in processes if p.is_running()))
                except psutil.Error:pass
                self.stop.wait(.1)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def __exit__(self,*exc):
        self.stop.set();self.thread.join();self.wall=time.perf_counter()-self.t0
        self.cpu_seconds=sum(self.proc.cpu_times()[:2])-self.cpu0
    def report(self):
        import torch
        return dict(wall_seconds=self.wall,cpu_process_seconds=self.cpu_seconds,
            peak_process_tree_rss_bytes_sampled=self.rss,rss_sampling_seconds=.1,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            caveat='RSS sampled, CUDA allocator peaks exclude driver/other processes; no energy/carbon measurement.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',default='data/triviaqa_val.jsonl')
    p.add_argument('--out',default='runs/test_rag');p.add_argument('--artifacts',default='artifacts');p.add_argument('--device',default='cuda')
    p.add_argument('--mode',choices=['probe','rag'],default='rag');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--original-threshold',type=float,default=.9);p.add_argument('--final-threshold',type=float,default=.75)
    p.add_argument('--exclusions',default='training_exclusions.json');p.add_argument('--allow-unverified-overlap',action='store_true')
    p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    if a.limit<0 or a.bootstrap<0:p.error('Nonnegative limit/bootstrap required')
    if not 0<=a.original_threshold<=1 or not 0<=a.final_threshold<=1:p.error('Thresholds must be [0,1]')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'records.jsonl').exists():raise RuntimeError('Use a fresh run directory; do not mix results/configurations')
    if not Path('pipeline_manifest.json').exists() or not Path('models.lock.json').exists():raise RuntimeError('Run prepare_repo.py first')
    #data=[json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
    data = []

    with Path(a.data).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record in {a.data}, "
                    f"physical line {line_number}, column {exc.colno}: "
                    f"{exc.msg}. "
                    "The dataset may be truncated or incorrectly formatted. "
                    "No records were skipped."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at physical line {line_number}"
                )

            data.append(record)
    exclusion=json.loads(Path(a.exclusions).read_text()) if Path(a.exclusions).exists() else None
    if exclusion is None and not a.allow_unverified_overlap:raise RuntimeError('Training exclusions missing. Supply prepared.csv during assembly or explicitly acknowledge unknown leakage.')
    ids=set(exclusion.get('ids',[])) if exclusion else set();questions=set(exclusion.get('normalized_questions',[])) if exclusion else set()
    eligible=[];removed=[];seen=set()
    for row in data:
        if not row.get('GT') or not row.get('aliases'):raise ValueError('Unlabeled data cannot be evaluated')
        key=normalize(row['question'])
        if row['id'] in ids or key in questions:removed.append(row['id']);continue
        if row['id'] in seen:raise ValueError('Duplicate benchmark question ID')
        seen.add(row['id']);eligible.append(row)
    if a.limit:eligible=eligible[:a.limit]
    if not eligible:raise ValueError('No eligible rows')
    manifest=dict(arguments=vars(a),dataset_sha256=sha(a.data),n_input=len(data),n_selected=len(eligible),
        removed_overlap_ids=removed,overlap_verified=exclusion is not None,question_ids=[r['id'] for r in eligible],
        platform=platform.platform(),python=platform.python_version(),pipeline=json.loads(Path('pipeline_manifest.json').read_text()),
        models=json.loads(Path('models.lock.json').read_text()),
        dataset_manifest=json.loads(Path(a.data).with_suffix('.manifest.json').read_text()) if Path(a.data).with_suffix('.manifest.json').exists() else None)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cache=Path(os.getenv('HF_HOME',str(Path.home()/'.cache/huggingface')))
    disk0=bytes_on_disk(cache);rows=[];lat=[];stages=[]
    import torch
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    from runtime import Runtime
    with Monitor() as monitor:
        runtime=Runtime(a.artifacts,a.device,a.original_threshold,a.final_threshold,a.mode)
        with (out/'records.jsonl').open('w') as log:
            for i,row in enumerate(eligible):
                start=time.perf_counter();error=None
                try:
                    result=runtime.answer(row['question']) # ONLY question, never GT/aliases
                    error=result.get('error')
                except Exception as exc:
                    result=dict(original_tag_output='',final_answer='abstain',retrieval_done=False,error=str(exc));error=str(exc)
                elapsed=time.perf_counter()-start;lat.append(elapsed)
                ma=result.get('original_tag_output','');pa=result.get('final_answer','abstain')
                record=dict(id=row['id'],TQ=row['question'],GT=row['GT'],aliases=row['aliases'],MA=ma,PA=pa,
                    MA_scores=score(ma,row['aliases']),PA_scores=score(pa,row['aliases']),seconds=elapsed,error=error,result=result)
                log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                summary_row={k:v for k,v in record.items() if k!='result'}
                summary_row['result']={'retrieval_done':result.get('retrieval_done',False)}
                rows.append(summary_row);stages.append(result.get('runtime',{}).get('stages',{}))
                print(f'{i+1}/{len(eligible)} seconds={elapsed:.2f} MA_EM={record["MA_scores"]["em"]} PA_EM={record["PA_scores"]["em"]}',flush=True)
    metrics=summarize(rows,a.bootstrap)
    metrics.update(complete=len(rows)==len(eligible),mode=a.mode,overlap_verified=exclusion is not None,
        results_claim_ready=all(not r.get('error') for r in rows) and exclusion is not None,
        scoring_warning='Strict EM penalizes full-sentence MA versus short-span PA. Compare token F1 and secondary alias-span too; formatting gains are not proof of hallucination reduction.')
    footprint=monitor.report();ordered=sorted(lat)
    footprint.update(n=len(rows),mean_query_seconds=statistics.mean(lat),median_query_seconds=statistics.median(lat),
        p95_query_seconds=ordered[int(.95*(len(ordered)-1))],first_query_seconds=lat[0],
        warm_mean_query_seconds=statistics.mean(lat[1:]) if len(lat)>1 else None,
        queries_per_second=len(rows)/monitor.wall,artifact_bytes=bytes_on_disk(a.artifacts),
        hf_cache_before_bytes=disk0,hf_cache_after_bytes=bytes_on_disk(cache),
        mean_original_generation_seconds=statistics.mean(x.get('original_generation_seconds',0.) for x in stages),
        mean_all_probe_likelihood_seconds=statistics.mean(x.get('probe_and_likelihood_seconds',0.) for x in stages),
        note='First query includes lazy model downloads/load when uncached. Total is measured combined MA+PA pipeline, not separate standalone MA throughput. Live Wikipedia/cache state affects latency.')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'footprint.json').write_text(json.dumps(footprint,indent=2))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['TQ','GT','MA','PA'])
        for r in rows:w.writerow([r['TQ'],r['GT'],r['MA'],r['PA']])
        w.writerow(['__METRICS__','alias-aware normalized exact match',f'M1={metrics["M1"]:.8f}',f'M2={metrics["M2"]:.8f}'])
    subprocess.run([os.sys.executable,'-m','pip','freeze'],stdout=(out/'environment.freeze.txt').open('w'),check=True)
    print(json.dumps(dict(metrics=metrics,footprint=footprint),indent=2))
if __name__=='__main__':main()
