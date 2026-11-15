"""AuditKIT integration over recorded predictions."""
import argparse,json
from pathlib import Path
from metrics import score

def audit(rows):
    import auditkit as ak
    byid={r['id']:r for r in rows}
    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def alias_em(sample,output):
        return score(str(output),byid[str(sample.input)]['aliases'])['em']
    samples=[ak.Sample(input=r['id'],target=r['GT']) for r in rows]
    report={}
    for name in ['MA','PA']:
        lookup={r['id']:r[name] for r in rows}
        result=ak.evaluate(samples,model=lambda prompts:[lookup[str(q)] for q in prompts],scorers=[alias_em])
        if getattr(result,'errors',None):raise RuntimeError(str(result.errors))
        report[name]=dict(headline=result.headline,summary=str(result.summary()))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.records).read_text().splitlines() if x.strip()]
    result=audit(rows);Path(a.output).write_text(json.dumps(result,indent=2,default=str));print('AuditKIT evaluation completed')
