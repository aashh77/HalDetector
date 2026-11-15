
import hashlib,importlib,json,os,threading,time
from pathlib import Path
from types import SimpleNamespace

class Runtime:
    def __init__(self,out='artifacts',device='cuda',original_threshold=.9,final_threshold=.75,mode='rag'):
        self.lock=threading.RLock()
        manifest=json.loads(Path('pipeline_manifest.json').read_text())
        for name,digest in {**manifest['source_hashes'],**manifest.get('release_hashes',{})}.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed after assembly: '+name)
        for name,digest in manifest['artifact_hashes'].items():
            if hashlib.sha256((Path(out)/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Artifact changed after assembly: '+name)
        if Path('generic_config.json').exists():os.environ['TAG_GENERIC_CONFIG']=str(Path('generic_config.json').resolve())
        self.tp=importlib.import_module('tag_rag_v3')
        self.out=out;self.device=device;self.original_threshold=original_threshold;self.final_threshold=final_threshold;self.mode=mode
        self.tag=None;self.pairs=None;self.timings={};self.loads={}
        self.model_lock=json.loads(Path('models.lock.json').read_text()) if Path('models.lock.json').exists() else {}
        self.original_TAG=self.tp.TAG;self.original_Pairs=self.tp.PairModels
        self.tp.TAG=self.get_tag;self.tp.PairModels=self.get_pairs
        for env,key in [('TAG_QA_REVISION','qa'),('TAG_NLI2_REVISION','nli2'),('TAG_EMBED_REVISION','embedding')]:
            if self.model_lock.get(key):os.environ[env]=self.model_lock[key]['revision']
    def get_tag(self,*a,**kw):
        if self.tag is None:
            start=time.perf_counter();self.tag=self.original_TAG(*a,**kw);self.loads['tag_seconds']=time.perf_counter()-start
            self._instrument(self.tag,'generate','original_generation_seconds')
            self._instrument(self.tag,'measure','probe_and_likelihood_seconds')
        return self.tag
    def get_pairs(self,*a,**kw):
        if self.pairs is None:
            start=time.perf_counter();self.pairs=self.original_Pairs(*a,**kw);self.loads['pair_models_seconds']=time.perf_counter()-start
        return self.pairs
    def _instrument(self,obj,name,label):
        original=getattr(obj,name)
        def measured(*a,**kw):
            self.sync();start=time.perf_counter()
            try:return original(*a,**kw)
            finally:
                self.sync();self.timings[label]=self.timings.get(label,0.)+time.perf_counter()-start
        setattr(obj,name,measured)
    @staticmethod
    def sync():
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.synchronize()
        except ImportError:pass
    def answer(self,question):
        if not isinstance(question,str) or not question.strip():raise ValueError('Nonempty question required')
        if len(question)>4000:raise ValueError('Question exceeds 4000-character UI/evaluation limit')
        with self.lock:
            self.timings={};self.sync();start=time.perf_counter()
            a=SimpleNamespace(out=self.out,device=self.device,question=question,
                original_threshold=self.original_threshold,final_threshold=self.final_threshold,
                probe_weight=.8,min_entailment=.65,min_relevance=.4,min_answer_relevance=.4,max_contradiction=.2,
                always_retrieve=False,respect_learned_policy=False,user_agent=os.getenv('WIKIMEDIA_USER_AGENT'),
                search_query=None,cache_dir=os.getenv('WIKI_CACHE_DIR','wiki_cache_v3'),max_pages=10,top_k=6,rag_attempts=2,
                rank_revision=self.model_lock.get('ranker',{}).get('revision','main'),
                nli_revision=self.model_lock.get('nli1',{}).get('revision','main'))
            if self.mode=='probe':
              
                import joblib
                b=joblib.load(Path(self.out)/'probe.joblib')
                for name in ['torch','transformers','scikit-learn']:
                    import importlib.metadata
                    expected=b.get('dependencies',{}).get(name)
                    if expected and importlib.metadata.version(name)!=expected:raise ValueError('Training version mismatch: '+name)
                tag=self.get_tag(b,self.device);prompt=self.tp.BASE_PROMPT.replace('{question}',question)
                text,count=tag.generate(prompt,100)
                try:measure=tag.measure(prompt,text)
                except Exception as exc:measure=dict(available=False,error=str(exc))
                score=self.tp.original_score(measure,.8);accepted=score is not None and score>=self.original_threshold and not self.tp.unknown(text)
                result=self.tp.blank_result(question,a)
                result.update(original_tag_output=text,original_probe=measure,
                    original_metrics=dict(overall_score=score,probe_weight=.8,formula='0.8*P+0.2*L'),
                    final_answer=text if accepted else 'abstain',final_score=score,
                    retrieval_done=False,action='answer_original' if accepted else 'abstain',
                    retrieval_reasons=[] if accepted else ['probe_only_ablation_withheld'])
            else:result=self.tp.run(a)
            self.sync();elapsed=time.perf_counter()-start
            result['runtime']=dict(total_seconds=elapsed,stages=dict(self.timings),model_loads=dict(self.loads),mode=self.mode)

            try:
                import generic_verifier as g
                if g.extra_models.cache_info().currsize:
                    m=g.extra_models()
                    for name in ['qa_cache','emb_cache','nli_cache']:
                        getattr(m,name,{}).clear()
            except (ImportError,AttributeError):pass
            return result
    def compact(self,result):
        from kit_workflow import compact_result
        return compact_result(result)
