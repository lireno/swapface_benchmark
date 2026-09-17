"""CPU-only regressions. Stub I3D validates wiring, not scientific model accuracy."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.linalg import sqrtm
import torch

from tools import fvd_paired as f

ROOT = Path(__file__).resolve().parents[1]


def video(path, n=20, fps=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (32,32))
    assert writer.isOpened()
    for i in range(n):
        writer.write(np.full((32,32,3),20+i*3,dtype=np.uint8))
    writer.release()


def test_short_shared_segment_not_relative_independent_start():
    g,o,w = f.paired_indices(81,25,300,25,15,'short',.975)
    assert g==o and w==81 and max(g)<81
    assert g[0]==64
    assert f.paired_indices(300,25,300,25,15,'short',1)[0]==list(range(66,81))


def test_long_and_fps_alignment():
    g,o,w = f.paired_indices(300,25,360,30,15,'long',1)
    assert w==300 and g==list(range(285,300))
    assert o==[round(i/25*30) for i in g]
    with pytest.raises(ValueError,match='cover'):
        f.paired_indices(300,25,81,25,15,'long',0)


def test_short_repeat_and_decode_errors(tmp_path):
    p=tmp_path/'x.mp4';video(p,4)
    g,o,_=f.paired_indices(4,25,4,25,15,'short',.9)
    a=f.decode_indices(p,g,15,(32,32))
    b=f.decode_indices(p,o,15,(32,32))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[0],a[4])
    with pytest.raises(RuntimeError,match='incomplete decode'):
        f.decode_indices(p,list(range(15)),15,(32,32))
    duplicate=f.decode_indices(p,[0,0,2],3,(32,32))
    np.testing.assert_array_equal(duplicate[0],duplicate[1])


def test_fvd_matches_reference_and_rejects_invalid():
    rng=np.random.default_rng(42)
    x,y=rng.normal(size=(30,8)),rng.normal(size=(30,8))
    a,b=np.cov(x,rowvar=False),np.cov(y,rowvar=False)
    delta=x.mean(0)-y.mean(0)
    expected=delta@delta+np.trace(a+b-2*sqrtm(a@b).real)
    assert f.fvd(x,y)==pytest.approx(expected,rel=1e-10)
    assert f.fvd(x,x)==pytest.approx(0,abs=1e-10)
    assert f.fvd(x,y)==pytest.approx(f.fvd(y,x),rel=1e-10)
    rank_deficient=rng.normal(size=(3,400))
    assert f.fvd(rank_deficient,rank_deficient)==pytest.approx(0,abs=1e-4)
    # Two-sample 400D covariance is rank one: compare its exact closed form.
    a,b=rng.normal(size=(2,400))*4,rng.normal(size=(2,400))*4
    u,v=(a[0]-a[1])/np.sqrt(2),(b[0]-b[1])/np.sqrt(2)
    delta=a.mean(0)-b.mean(0)
    exact=delta@delta+u@u+v@v-2*abs(u@v)
    assert f.fvd(a,b)==pytest.approx(exact,rel=1e-12)
    for invalid in (x[:1],np.full_like(x,np.nan)):
        with pytest.raises(ValueError): f.fvd(invalid,y)


def test_cache_fingerprint_tracks_video_and_indices(tmp_path):
    p=tmp_path/'x.mp4';p.write_bytes(b'old')
    items=[{'case_id':'x','origin_path':str(p),'origin_indices':[0,1]}]
    a=f.cache_payload(items,'origin',{'version':2})
    p.write_bytes(b'changed')
    assert a!=f.cache_payload(items,'origin',{'version':2})
    a=f.cache_payload(items,'origin',{'version':2})
    items[0]['origin_indices']=[1,2]
    assert a!=f.cache_payload(items,'origin',{'version':2})


def test_cache_validates_ids_count_and_nan(tmp_path):
    p=tmp_path/'cache.npz'
    def write(values,ids):
        f.atomic_write(p,lambda file:np.savez(file,activations=values,case_ids=ids,fingerprint='key'))
    write(np.zeros((2,400)),['a','b'])
    assert f.cached_activations(p,'key',['a','b']).shape==(2,400)
    assert f.cached_activations(p,'key',['b','a']) is None
    assert f.cached_activations(p,'wrong',['a','b']) is None
    write(np.zeros((1,400)),['a','b'])
    assert f.cached_activations(p,'key',['a','b']) is None
    write(np.full((2,400),np.nan),['a','b'])
    assert f.cached_activations(p,'key',['a','b']) is None
    p.write_bytes(b'PK\x03\x04truncated')
    assert f.cached_activations(p,'key',['a','b']) is None


def test_feature_merge_preserves_order_and_rejects_missing_duplicates():
    a,b=np.ones((1,400)),np.zeros((1,400))
    rows=f.merge_feature_shards([(['b'],b),(['a'],a)],['a','b'])
    np.testing.assert_array_equal(rows,np.vstack([a,b]))
    with pytest.raises(ValueError,match='duplicate'):
        f.merge_feature_shards([(['a'],a),(['a'],b)],['a','b'])
    with pytest.raises(ValueError,match='missing'):
        f.merge_feature_shards([(['a'],a)],['a','b'])


@pytest.fixture
def stub_run(tmp_path,monkeypatch):
    p=tmp_path/'x.mp4';video(p)
    m=tmp_path/'mapping.json'
    m.write_text(json.dumps([{'case_id':c,'generated':str(p),'ref_video':str(p)} for c in ['a','b']]))
    code=tmp_path/'pytorch_i3d_model';code.mkdir()
    (code/'pytorch_i3d.py').write_text('# test stub\n')
    weights=tmp_path/'weights.pt';torch.save({},weights)
    class Stub(torch.nn.Module):
        def __init__(self,*args,**kwargs): super().__init__()
        def forward(self,x):
            return x.mean((1,2,3,4))[:,None,None].expand(-1,400,1)
    monkeypatch.setitem(sys.modules,'pytorch_i3d_model.pytorch_i3d',SimpleNamespace(InceptionI3d=Stub))
    return SimpleNamespace(mapping=m,manifest=None,limit=0,seed=42,benchmark_mode='short',
        video_length=15,decode_width=32,decode_height=32,batch_size=2,device='cpu',
        i3d_root=tmp_path,weights=weights,output=tmp_path/'fvd.json',cache_dir=tmp_path/'cache',no_cache=False)


@pytest.mark.parametrize('mode',['short','long'])
def test_end_to_end_cache_resume_and_no_cache(stub_run,monkeypatch,mode):
    a=stub_run;a.benchmark_mode=mode
    f.run(a)
    result=json.loads(a.output.read_text())
    assert result['case_count']==2 and result['status']=='complete' and result['fvd']==0
    assert len(list(a.cache_dir.glob('*.npz')))==2
    def fail(*args): raise AssertionError('must load complete cache')
    monkeypatch.setattr(f,'extract_activations',fail)
    f.run(a)
    a.no_cache=True
    with pytest.raises(AssertionError,match='complete cache'): f.run(a)


def test_failed_side_not_cached(stub_run,monkeypatch):
    original=f.extract_activations
    def extract(items,side,model,args):
        if side=='generated': raise RuntimeError('decode failed')
        return original(items,side,model,args)
    monkeypatch.setattr(f,'extract_activations',extract)
    with pytest.raises(RuntimeError,match='decode failed'): f.run(stub_run)
    assert not list(stub_run.cache_dir.glob('generated*'))
    assert not stub_run.output.exists()
    monkeypatch.setattr(f,'extract_activations',original)
    f.run(stub_run)
    assert json.loads(stub_run.output.read_text())['case_count']==2


def test_only_fvd_needs_no_reference_or_roi_and_summary(tmp_path):
    for c in ('a','b'): video(tmp_path/f'{c}.mp4')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':c,'origin_video':f'{c}.mp4'} for c in ('a','b')]}))
    mapping=tmp_path/'mapping.json'
    def call(script,*args):
        subprocess.run([sys.executable,str(ROOT/'tools'/script),*map(str,args)],check=True,capture_output=True)
    call('prepare_results.py','--manifest',manifest,'--results-dir',tmp_path,'--output',mapping,'--no-reference','--no-roi')
    report=tmp_path/'report.json'
    call('validate_mapping.py','--mapping',mapping,'--output',report,'--errors',tmp_path/'errors','--no-roi')
    artifact=tmp_path/'fvd.json'
    artifact.write_text(json.dumps({'fvd':1.23,'case_count':2,'failure_count':0,'protocol':{'aggregation':'dataset-level'}}))
    summary=tmp_path/'summary.json'
    call('summarize.py','--fvd',artifact,'--selected','fvd','--input-report',report,'--output',summary)
    data=json.loads(summary.read_text())
    assert data['metrics_flat']=={'fvd':1.23}
    assert data['protocol']['fvd']['aggregation']=='dataset-level'
    artifact.write_text(json.dumps({'fvd':None,'failure_count':1,'status':'failed'}))
    result=subprocess.run([sys.executable,str(ROOT/'tools/summarize.py'),'--selected','fvd','--fvd',str(artifact),'--output',str(summary)],capture_output=True)
    assert result.returncode==1


@pytest.mark.parametrize('mode',['short','long'])
def test_public_shell_fvd_only_and_all_resume(tmp_path,mode):
    # Real shell, prepare/validate, evaluator and summary; only the network and
    # device are stubbed so this test needs neither checkpoint nor GPU.
    code=tmp_path/'pytorch_i3d_model';code.mkdir()
    (code/'__init__.py').write_text('')
    (code/'pytorch_i3d.py').write_text('''import torch
class InceptionI3d(torch.nn.Module):
    def __init__(self,*args,**kwargs): super().__init__()
    def forward(self,x): return x.mean((1,2,3,4))[:,None,None].expand(-1,400,1)
''')
    weights=tmp_path/'test_weights.pt';torch.save({},weights)
    results=tmp_path/'videos';results.mkdir()
    for c in ('a','b'): video(results/f'{c}.mp4',90)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':c,'origin_video':str(results/f'{c}.mp4')} for c in ('a','b')]}))
    wrapper=tmp_path/'python_cpu_test'
    wrapper.write_text(f'''#!{sys.executable}
import os,sys
args=sys.argv[1:]
if args and args[0].endswith('/eval_fvd_streaming.py'):
    args[args.index('--device')+1]='cpu'
os.execv(sys.executable,[sys.executable]+args)
''')
    wrapper.chmod(0o755)
    out=tmp_path/'evaluation'
    cmd=['bash',str(ROOT/'scripts/evaluate.sh'),str(results),'--manifest',str(manifest),
         '--benchmark-mode',mode,'--metrics','fvd','--output-dir',str(out),
         '--model-profile','assets','--models-root',str(tmp_path/'unused_models'),
         '--fvd-i3d-root',str(tmp_path),'--fvd-weights',str(weights),'--gpu-list','0,1']
    env={**os.environ,'PYTHON_BIN':str(wrapper),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    first=subprocess.run(cmd,env=env,capture_output=True,text=True)
    assert first.returncode==0,first.stdout+first.stderr
    data=json.loads((out/'summary.json').read_text())
    assert data['metrics_flat']=={'fvd':0.0}
    assert data['protocol']['fvd']['benchmark_mode']==mode
    assert not any(data['failure_count'].values())
    artifact=json.loads((out/'fvd.json').read_text())
    assert artifact['execution']['worker_count']==2
    assert artifact['execution']['gpu_list']==['0','1']
    assert artifact['activation_shape']==[2,400]
    assert not (out/'identity_strict.json').exists()
    before=(out/'fvd.json').stat().st_mtime_ns
    resumed=subprocess.run(cmd,env=env,capture_output=True,text=True)
    assert resumed.returncode==0 and '[resume] fvd' in resumed.stdout
    assert (out/'fvd.json').stat().st_mtime_ns==before
    # Verify all expands to FVD too, without starting any unrelated metric.
    all_cmd=cmd.copy();all_cmd[all_cmd.index('--metrics')+1]='all'
    all_cmd+=['--exclude-metrics','identity,facebench,vbench','--no-resume']
    rerun=subprocess.run(all_cmd,env=env,capture_output=True,text=True)
    assert rerun.returncode==0,rerun.stdout+rerun.stderr
    assert '[run] fvd' in rerun.stdout
    assert (out/'fvd.json.previous').exists()
    assert (out/'fvd.json').stat().st_mtime_ns!=before
