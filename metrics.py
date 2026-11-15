"""Alias-aware normalized EM and token F1; no LLM judge, no gold in inference."""
from collections import Counter
import re,string

def normalize(s):
    s=str(s).lower().replace('_',' ')
    s=''.join(' ' if c in string.punctuation else c for c in s)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def token_f1(a,b):
    a=normalize(a).split();b=normalize(b).split()
    if not a or not b:return float(a==b)
    common=sum((Counter(a)&Counter(b)).values())
    return 2*common/(len(a)+len(b))
def score(pred,aliases):
    gold=[normalize(x) for x in aliases if normalize(x)]
    if not gold:raise ValueError('No nonempty ground truth aliases')
    p=normalize(pred)
    # Span metric is secondary: a correct name inside a false sentence can pass.
    return dict(em=float(p in gold),f1=max(token_f1(pred,x) for x in aliases),
        alias_span=float(any((' '+g+' ') in (' '+p+' ') for g in gold)))
def abstains(text):return not str(text).strip() or str(text).strip().lower()=='abstain'
def summarize(rows,bootstrap=2000,seed=42):
    import random
    n=len(rows)
    if not n:raise ValueError('No evaluated rows')
    m1=sum(x['MA_scores']['em'] for x in rows)/n;m2=sum(x['PA_scores']['em'] for x in rows)/n
    delta=[x['PA_scores']['em']-x['MA_scores']['em'] for x in rows]
    rng=random.Random(seed)
    boot=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(bootstrap)) if bootstrap else []
    answered=[x for x in rows if not abstains(x['PA'])]
    wrong=sum(x['PA_scores']['em']==0 for x in answered)
    # Benchmark nonmatch != clinically or factually established hallucination.
    return dict(n=n,M1=m1,M2=m2,delta_percentage_points=100*(m2-m1),
        paired_bootstrap_delta_95ci=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]] if boot else None,
        baseline_f1=sum(x['MA_scores']['f1'] for x in rows)/n,
        pipeline_f1=sum(x['PA_scores']['f1'] for x in rows)/n,
        baseline_alias_span=sum(x['MA_scores']['alias_span'] for x in rows)/n,
        pipeline_alias_span=sum(x['PA_scores']['alias_span'] for x in rows)/n,
        coverage=len(answered)/n,selective_em_error=wrong/len(answered) if answered else None,
        baseline_nonmatch_rate=1-m1,pipeline_wrong_delivered_per_question=wrong/n,
        abstention_rate=1-len(answered)/n,
        fixed=sum(x['MA_scores']['em']==0 and x['PA_scores']['em']==1 for x in rows),
        regressed=sum(x['MA_scores']['em']==1 and x['PA_scores']['em']==0 for x in rows),
        retrieval_rate=sum(bool(x['result'].get('retrieval_done')) for x in rows)/n,
        errors=sum(bool(x.get('error')) for x in rows),
        interpretation='M1/M2 are alias-aware normalized exact match, not semantic truth. Nonmatches and abstentions must not all be called hallucinations. Improvement is an empirical outcome, not guaranteed.')
