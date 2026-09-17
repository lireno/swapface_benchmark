"""Timestamp-paired, memory-bounded HiFiVFS 400-logit FVD. No per-case FVD.

Short: first 81 generated frames. Long: full generated timeline. Pick one
15-frame clip per case and read origin at the SAME timestamps. Framewise
metric stride does not apply here. Models/videos must use aligned time zero.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import zipfile
import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate

PROTOCOL_VERSION = "paired_timestamp_i3d_logits400_v2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--mapping", type=Path)
    source.add_argument("--manifest", type=Path, help="Legacy standalone manifest")
    p.add_argument("--origin-dir", type=Path)
    p.add_argument("--generated-dir", type=Path)
    p.add_argument("--benchmark-mode", choices=("short", "long"), default="short")
    p.add_argument("--i3d-root", type=Path, default=Path(__file__).resolve().parents[1]/"vendor")
    for key in ("weights", "output", "cache-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    for key, default in (("video-length",15), ("decode-width",640), ("decode-height",480),
                         ("batch-size",4), ("seed",42), ("limit",0)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpu-list", help="Physical GPU IDs; one persistent feature worker per GPU")
    p.add_argument("--no-cache", action="store_true", help="Bypass feature reads and writes")
    a = p.parse_args()
    if a.video_length < 15 or min(a.batch_size,a.decode_width,a.decode_height) <= 0 or a.limit < 0:
        p.error("video-length >=15; batch/size positive; limit nonnegative required")
    if a.manifest and not a.generated_dir:
        p.error("--manifest requires --generated-dir")
    return a


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_stamp(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path":str(path), "size":stat.st_size, "mtime_ns":stat.st_mtime_ns}


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid frames/FPS: {path}")
        return count, fps
    finally:
        cap.release()


def deterministic_fraction(case_id, seed):
    payload = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
    return int.from_bytes(payload[:8], "big") / float(2**64-1)


def paired_indices(ng, fg, no, fo, length, mode, fraction):
    """Require full origin coverage. round/ties-to-even matches public evaluator."""
    if mode not in ("short","long") or min(ng,no,length) <= 0:
        raise ValueError("invalid mode/count/length")
    if not all(math.isfinite(f) and f > 0 for f in (fg,fo)) or not 0 <= fraction <= 1:
        raise ValueError("invalid FPS/fraction")
    window = min(81,ng) if mode == "short" else ng
    if round((window-1)/fg*fo) >= no:
        raise ValueError("origin does not cover generated evaluation window")
    start = int(round(fraction*max(window-length,0)))
    g = list(range(start,start+min(window,length)))
    o = [round(i/fg*fo) for i in g]
    return g,o,window


def decode_indices(path, indices, length, size):
    """Sequential decode avoids imprecise seeks; no padding on decode failure."""
    if not indices or len(indices)>length or indices != sorted(indices) or min(indices)<0:
        raise ValueError("invalid sampled indices")
    cap = cv2.VideoCapture(str(path))
    frames, needed = {}, set(indices)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {path}")
        for i in range(max(indices)+1):
            if i in needed:
                ok,frame = cap.read()
                if not ok:
                    raise RuntimeError(f"incomplete decode: {path}, frame={i}")
                frames[i] = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),size,
                                       interpolation=cv2.INTER_LINEAR)
            elif not cap.grab():
                raise RuntimeError(f"incomplete decode: {path}, frame={i}")
    finally:
        cap.release()
    # Only a planned short window is repeated, identically for both sides.
    return np.stack([frames[indices[i%len(indices)]] for i in range(length)]).astype(np.float32)


def preprocess(batch):
    videos = batch.permute(0,4,1,2,3)
    videos = interpolate(videos,size=[videos.shape[2],224,224],mode="trilinear",align_corners=False)
    return 2*videos/255.0-1


def validate_activations(values,count):
    if values.shape != (count,400) or not np.isfinite(values).all():
        raise ValueError(f"invalid activations: expected {(count,400)}, got {values.shape}")


def extract_activations(items,side,model,args):
    activations = []
    for offset in range(0,len(items),args.batch_size):
        chunk = items[offset:offset+args.batch_size]
        videos = []
        for item in chunk:
            try:
                videos.append(decode_indices(item[f"{side}_path"],item[f"{side}_indices"],
                    args.video_length,(args.decode_width,args.decode_height)))
            except Exception as error:
                raise RuntimeError(f"case={item['case_id']} side={side}: {error}") from error
        tensor = preprocess(torch.from_numpy(np.stack(videos))).to(args.device)
        with torch.inference_mode():
            output = model(tensor).mean(dim=-1).detach().cpu().numpy()
        validate_activations(output,len(chunk))
        activations.append(output)
        print(f"[{side}] {offset+len(chunk)}/{len(items)}",flush=True)
    result = np.vstack(activations)
    validate_activations(result,len(items))
    return result


def fvd(first,second):
    """Float64 Bures distance via covariance factors, stable for N << 400.

    With centered/scaled features A and B, covariances are A.T@A and B.T@B.
    trace(sqrt(sqrt(C1) C2 sqrt(C1))) equals nuclear_norm(A@B.T).
    SVD avoids sqrtm's complex/rank-deficient instability and amplification
    of roundoff-sized positive eigenvalues by square roots.
    """
    first,second = np.asarray(first,dtype=np.float64),np.asarray(second,dtype=np.float64)
    if (first.ndim!=2 or second.ndim!=2 or min(len(first),len(second))<2
            or first.shape[1]!=second.shape[1] or first.shape[1]<1
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("FVD requires >=2 finite samples per set with matching feature dimensions")
    mean1,mean2 = first.mean(0),second.mean(0)
    a = (first-mean1)/np.sqrt(len(first)-1)
    b = (second-mean2)/np.sqrt(len(second)-1)
    delta = mean1-mean2
    scale = float(delta@delta+np.sum(a*a)+np.sum(b*b))
    cross = float(np.linalg.svd(a@b.T,compute_uv=False).sum())
    value = scale-2*cross
    if not math.isfinite(value) or value < -1e-10*max(scale,1.):
        raise ValueError(f"invalid FVD: {value}")
    return max(value,0.)


def atomic_write(path,writer):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+".",delete=False) as file:
        tmp = Path(file.name)
        try:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
            os.replace(tmp,path)
        finally:
            tmp.unlink(missing_ok=True)


def write_json(path,payload):
    atomic_write(path,lambda file:file.write((json.dumps(payload,indent=2,allow_nan=False)+"\n").encode()))


def cache_payload(items,side,protocol):
    return {"protocol":protocol,"side":side,"items":[
        {"case_id":i["case_id"],"asset":asset_stamp(i[f"{side}_path"]),
         "indices":i[f"{side}_indices"]} for i in items]}


def cached_activations(path,fingerprint,case_ids):
    try:
        with np.load(path,allow_pickle=False) as data:
            if data["fingerprint"].item()!=fingerprint or data["case_ids"].tolist()!=case_ids:
                raise ValueError("cache provenance mismatch")
            values = data["activations"]
            validate_activations(values,len(case_ids))
            return values
    except (ValueError,OSError,KeyError,EOFError,zipfile.BadZipFile) as error:
        print(f"[cache miss] {path}: {error}",flush=True)
        return None


def load_rows(args):
    if args.mapping:
        rows = json.loads(args.mapping.read_text())
    else:
        try:
            from tools.prepare_results import find_generated,resolve_asset
        except ModuleNotFoundError:
            from prepare_results import find_generated,resolve_asset
        rows = []
        cases = json.loads(args.manifest.read_text())["cases"]
        if args.limit:
            cases = cases[:args.limit]
        for case in cases:
            cid = case["case_id"]
            origin = args.origin_dir/f"{cid}.mp4" if args.origin_dir else resolve_asset(args.manifest,case["origin_video"])
            rows.append({"case_id":cid,"ref_video":str(origin),
                         "generated":str(find_generated(args.generated_dir,cid))})
    if args.limit:
        rows = rows[:args.limit]
    ids = [r.get("case_id",r.get("name")) for r in rows]
    if len(rows)<2 or any(not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError("FVD requires >=2 distinct, named cases")
    return rows


def load_model(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sys.path.insert(0,str(Path(args.i3d_root).resolve()))
    from pytorch_i3d_model.pytorch_i3d import InceptionI3d
    model = InceptionI3d(400,in_channels=3)
    model.load_state_dict(torch.load(args.weights,map_location="cpu",weights_only=True))
    return model.eval().to(args.device)


def feature_spec(items,side,protocol,cache_dir):
    identity=cache_payload(items,side,protocol)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return identity,digest,Path(cache_dir)/f"{side}_{digest}.npz"


def save_features(path,values,ids,digest):
    validate_activations(values,len(ids))
    atomic_write(path,lambda file:np.savez(file,activations=values,
                 case_ids=np.asarray(ids),fingerprint=np.asarray(digest)))


def merge_feature_shards(shards,order):
    """Merge feature rows by case ID, never merge/average worker FVD scores."""
    rows={}
    for ids,values in shards:
        validate_activations(values,len(ids))
        for cid,value in zip(ids,values):
            if cid in rows:
                raise ValueError(f"duplicate feature case: {cid}")
            rows[cid]=value
    if set(rows)!=set(order) or len(order)!=len(set(order)):
        raise ValueError("missing/unexpected feature cases")
    return np.stack([rows[cid] for cid in order])


def parallel_features(items,sides,protocol,args):
    """Each GPU loads I3D once and handles BOTH sides of its case shard.

    Completed shard caches survive another worker's failure; retries only
    extract missing shards. No-cache uses a fresh run directory instead.
    """
    gpus=[g.strip() for g in args.gpu_list.split(',') if g.strip()]
    if not gpus or len(gpus)!=len(set(gpus)):
        raise ValueError("GPU list must contain distinct nonempty device IDs")
    gpus=gpus[:len(items)]
    run_root=args.output.parent/'fvd_workers'
    run_root.mkdir(parents=True,exist_ok=True)
    run_dir=Path(tempfile.mkdtemp(prefix='run_',dir=run_root))
    shard_root=run_dir/'uncached_features' if args.no_cache else args.cache_dir/'shards'
    inventory={side:[] for side in sides}
    processes=[]
    try:
        for rank,gpu in enumerate(gpus):
            start,end=len(items)*rank//len(gpus),len(items)*(rank+1)//len(gpus)
            subset=items[start:end]
            ids=[i['case_id'] for i in subset]
            jobs=[]
            for side in sides:
                identity,digest,path=feature_spec(subset,side,protocol,shard_root)
                values=cached_activations(path,digest,ids) if not args.no_cache and path.is_file() else None
                inventory[side].append((ids,digest,path))
                if values is None:
                    jobs.append({'side':side,'identity':identity,'fingerprint':digest,'output':str(path)})
            if not jobs:
                print(f'[fvd-shard-resume] rank={rank} cases=[{start},{end})',flush=True)
                continue
            request={'items':subset,'jobs':jobs,'protocol':protocol,'args':{
                'i3d_root':str(args.i3d_root.resolve()),'weights':str(args.weights.resolve()),
                'device':args.device if args.device=='cpu' else 'cuda:0',
                'video_length':args.video_length,'decode_width':args.decode_width,
                'decode_height':args.decode_height,'batch_size':args.batch_size,'seed':args.seed}}
            request_path=run_dir/f'worker_{rank}.json'
            write_json(request_path,request)
            log_path=run_dir/f'worker_{rank}_gpu{gpu}.log'
            with log_path.open('w') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('fvd_feature_worker.py')),
                    '--request',str(request_path)],env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},
                    stdout=log,stderr=subprocess.STDOUT)
            processes.append((rank,gpu,process,log_path))
            print(f'[fvd-shard] rank={rank} physical_gpu={gpu} cases=[{start},{end}) log={log_path}',flush=True)
        failures=[]
        for rank,gpu,process,log_path in processes:
            code=process.wait()
            if code:
                failures.append({'rank':rank,'gpu':gpu,'exit_code':code,'log':str(log_path)})
        if failures:
            raise RuntimeError(f'FVD feature workers failed: {failures}')
    finally:
        # Only our own unfinished children, e.g. if spawning was interrupted.
        for _,_,process,_ in processes:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
    merged={}
    for side,entries in inventory.items():
        shards=[]
        for ids,digest,path in entries:
            values=cached_activations(path,digest,ids)
            if values is None:
                raise RuntimeError(f'missing/invalid completed shard: {path}')
            shards.append((ids,values))
        merged[side]=merge_feature_shards(shards,[i['case_id'] for i in items])
    return merged,{'gpu_list':gpus,'worker_count':len(processes),'run_dir':str(run_dir),
                   'aggregation':'all feature rows gathered, one global FVD'}


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    items = []
    for row in load_rows(args):
        cid = row.get("case_id",row.get("name"))
        origin,generated = Path(row["ref_video"]).resolve(),Path(row["generated"]).resolve()
        no,fo = video_info(origin)
        ng,fg = video_info(generated)
        g,o,window = paired_indices(ng,fg,no,fo,args.video_length,args.benchmark_mode,
                                   deterministic_fraction(cid,args.seed))
        items.append({"case_id":cid,"origin_path":str(origin),"generated_path":str(generated),
            "origin_frames":no,"generated_frames":ng,"origin_fps":fo,"generated_fps":fg,
            "origin_indices":o,"generated_indices":g,"window_frames":window,
            "repeated_frames":max(args.video_length-len(g),0)})
    source_files = sorted((args.i3d_root/"pytorch_i3d_model").rglob("*.py"))
    if not source_files:
        raise FileNotFoundError("--i3d-root must contain pytorch_i3d_model/*.py")
    protocol = {"version":PROTOCOL_VERSION,"benchmark_mode":args.benchmark_mode,
        "reference_implementation":"HiFiVFS_wan/frechet_video_distance.py",
        "i3d_output":"400 Kinetics logits averaged over output time",
        "weights_sha256":sha256(args.weights),"evaluator_sha256":sha256(Path(__file__)),
        "feature_worker_sha256":sha256(Path(__file__).with_name('fvd_feature_worker.py')),
        "i3d_code_sha256":{str(p.relative_to(args.i3d_root)):sha256(p) for p in source_files},
        "video_length":args.video_length,"decode_size":[args.decode_width,args.decode_height],
        "i3d_input_size":[224,224],"normalization":"2*x/255-1","seed":args.seed,
        "segment_selection":"one case-hash start on generated window; origin nearest timestamp (round)",
        "short_clip_policy":"repeat both paired sequences only for genuine short windows",
        "spatial_scope":"full frame, no ROI","frame_stride":1,
        "aggregation":"dataset-level Frechet distance; not average per-case FVD",
        "video_fingerprint":"resolved path, size, mtime_ns; --no-cache for stat-preserving edits"}
    ids = [i["case_id"] for i in items]
    features,caches,specs = {},{},{}
    for side in ("origin","generated"):
        identity,digest,path=feature_spec(items,side,protocol,args.cache_dir)
        specs[side]=(identity,digest,path)
        values=cached_activations(path,digest,ids) if not args.no_cache and path.is_file() else None
        if values is not None:
            features[side]=values
            print(f"[resume] {path}",flush=True)
        caches[side]=str(path) if not args.no_cache else None
    missing=[s for s in specs if s not in features]
    execution={"gpu_list":None,"worker_count":0,"aggregation":"one global FVD"}
    if missing:
        if getattr(args,"gpu_list",None):
            extracted,execution=parallel_features(items,missing,protocol,args)
        else:
            model=load_model(args)
            extracted={s:extract_activations(items,s,model,args) for s in missing}
        for side in missing:
            identity,digest,path=specs[side]
            if cache_payload(items,side,protocol)!=identity:
                raise RuntimeError(f"{side} inputs changed during extraction")
            features[side]=extracted[side]
            if not args.no_cache:
                save_features(path,features[side],ids,digest)
    score = fvd(features["origin"],features["generated"])
    payload = {"status":"complete","fvd":score,"case_count":len(items),"failure_count":0,
        "activation_shape":list(features["origin"].shape),"protocol":protocol,
        "mapping":items,"failures":[],"cache":caches,"execution":execution}
    write_json(args.output,payload)
    print(json.dumps({"fvd":score,"case_count":len(items),"output":str(args.output)}))


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        write_json(args.output,{"status":"failed","fvd":None,"failure_count":1,
                                "failures":[{"error":str(error)}]})
        raise
    return 0
