#!/usr/bin/env python3
"""Two real cases per mode through the public entry; smoke, not benchmark scores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True,
                   help='Existing short200/long200 directories with source.json and results/')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu',default='4')
    p.add_argument('--models-root',type=Path,default=Path('/mnt/cpfs/users/lzk/modelscope_swapface_models/models'))
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    results=[]
    for mode in ('short','long'):
        source=a.source_root/(mode+'200')
        dest=a.output_root/mode
        dest.mkdir(parents=True,exist_ok=True)
        rows=json.loads((source/'source.json').read_text())['cases'][:2]
        manifest=dest/'manifest.json'
        manifest.write_text(json.dumps({'cases':[
            {'case_id':r['case_id'],'origin_video':r.get('origin_video') or r['ref_video']}
            for r in rows]},indent=2)+'\n')
        cmd=['bash',str(root/'scripts/evaluate.sh'),str(source/'results'),
             '--manifest',str(manifest),'--benchmark-mode',mode,'--metrics','fvd',
             '--output-dir',str(dest),'--gpu-list',a.gpu,'--model-profile','assets',
             '--models-root',str(a.models_root),'--fvd-batch-size','1']
        print('[smoke]',mode,flush=True)
        result=subprocess.run(cmd,env={**os.environ,'PYTHON_BIN':sys.executable})
        results.append({'mode':mode,'exit_code':result.returncode,'output':str(dest)})
    (a.output_root/'smoke_status.json').write_text(json.dumps(results,indent=2)+'\n')
    return int(any(r['exit_code'] for r in results))


if __name__=='__main__':
    raise SystemExit(main())
