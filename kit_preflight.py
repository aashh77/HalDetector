"""REAL SDK contract tests; no model weights/API keys. Required before enabling KITs.
Unlike test_two_kits.py, this must execute the installed third-party packages.
"""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from kit_adapters import curate_evidence,audit_values,package_info

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-lock',action='store_true');a=p.parse_args()
    rows=[dict(id='S1',passage='Fixture subject is on River Alpha.',relevance=.8),
          dict(id='S2',passage='Fixture subject is on River Alpha.',relevance=.9),
          dict(id='S3',passage='Fixture subject is NOT on River Alpha.',relevance=.95)]
    kept,curation=curate_evidence(rows)
    if {r['id'] for r in kept}!={'S2','S3'}:raise RuntimeError('CuratorKIT contract failed')
    values,audit=audit_values('Fixture question',dict(fixture=True,value=.75),dict(evidence_index=.75,source_eligible=1.),
        calculators=dict(evidence_index=lambda p:p['value'],source_eligible=lambda p:float(p['fixture'])))
    if values!={'evidence_index':.75,'source_eligible':1.}:raise RuntimeError('AuditKIT contract failed')
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        packages={name:package_info(name) for name in ['curatorkit','auditkit']},
        real_sdk_contracts_passed=True,curation=curation,audit=audit,
        note='Fixture SDK contract test, NOT factual benchmark performance or live model validation.')
    if a.write_lock:
        path=Path(__file__).with_name('kits.lock.json')
        if path.exists():
            backup=path.with_name('kits.lock.previous.'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
            backup.write_bytes(path.read_bytes())
        path.write_text(json.dumps(report,indent=2))
        print('Real SDK contracts passed; lock written:',path)
    else:print(json.dumps(report,indent=2))
if __name__=='__main__':main()
