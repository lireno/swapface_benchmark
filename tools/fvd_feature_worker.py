#!/usr/bin/env python3
"""Internal FVD feature worker. Never computes an independent FVD score."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import fvd_paired as f


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    request=json.loads(args.request.read_text())
    a=SimpleNamespace(**request['args'])
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    model=f.load_model(a)
    items=request['items'];ids=[i['case_id'] for i in items]
    for job in request['jobs']:
        side=job['side']
        if f.cache_payload(items,side,request['protocol'])!=job['identity']:
            raise RuntimeError(f'{side} input changed before extraction')
        values=f.extract_activations(items,side,model,a)
        if f.cache_payload(items,side,request['protocol'])!=job['identity']:
            raise RuntimeError(f'{side} input changed during extraction')
        f.save_features(job['output'],values,ids,job['fingerprint'])
    print('[fvd-worker] complete',flush=True)


if __name__=='__main__':
    main()
