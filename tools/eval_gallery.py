#!/usr/bin/env python3
"""Register benchmark evaluations and build the short/long static galleries."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "registries"
REPORT_ROOT = ROOT / "web_reports"

METRICS = {
    "id_sim": ("ID Sim", True, 4), "input_leak": ("Input leak", False, 4),
    "id_arc": ("ID Arc", True, 4), "id_ins": ("ID Ins", True, 4),
    "id_cur": ("ID Cur", True, 4), "id_variance": ("ID variance", False, 5),
    "face_detection_rate": ("Face detection", True, 4),
    "face_similarity": ("Face similarity", True, 4),
    "face_similarity_src": ("Input face similarity", False, 4),
    "pose_distance": ("Pose", False, 4), "gaze_l2_distance": ("Gaze L2", False, 4),
    "gaze_cosine_similarity": ("Gaze cosine", True, 4),
    "exp_l2_distance": ("Expression", False, 4),
    "gamma_l2_distance": ("Lighting", False, 4),
    "vbench_imaging_quality": ("Imaging quality", True, 4),
    "raw_musiq_spaq": ("MUSIQ-SPAQ", True, 4),
    "vbench_subject_consistency": ("Subject consistency", True, 4),
    "vbench_temporal_flickering": ("Temporal flickering", True, 4),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registry_path(mode: str) -> Path:
    return REGISTRY_DIR / f"{mode}.json"


def validate_mode(mode: str) -> None:
    if mode not in {"short", "long"}:
        raise ValueError(f"benchmark mode must be short or long, got {mode!r}")


def new_registry(mode: str, manifest: Path) -> dict[str, Any]:
    validate_mode(mode)
    manifest = manifest.resolve()
    payload = load_json(manifest)
    return {
        "schema_version": 1, "benchmark_mode": mode,
        "manifest": manifest.as_posix(), "manifest_sha256": sha256(manifest),
        "case_count": len(payload.get("cases", [])), "runs": [],
    }


@contextlib.contextmanager
def registry_lock(mode: str):
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = REGISTRY_DIR / f".{mode}.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def safe_id(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    if not result:
        raise ValueError("run id is empty after normalization")
    return result


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink report media: {link}")
    link.symlink_to(target.resolve())


def metric_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("mean", "value", "score"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
    return None


def case_metrics(evaluation_dir: Path, summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Merge per-case values from metric artifacts that expose case records."""
    merged: dict[str, dict[str, float]] = {}
    for artifact in summary.get("artifacts", {}).values():
        path = Path(artifact)
        if not path.is_file() or path.name == "input_report.json":
            continue
        payload = load_json(path)
        cases = payload.get("cases")
        if isinstance(cases, dict):
            rows = [(str(key), value) for key, value in cases.items()]
        elif isinstance(cases, list):
            rows = [(str(value.get("case_id", value.get("name", ""))), value) for value in cases]
        else:
            continue
        for case_id, row in rows:
            if not case_id or not isinstance(row, dict):
                continue
            destination = merged.setdefault(case_id, {})
            for key in METRICS:
                value = metric_number(row.get(key))
                if value is not None:
                    destination[key] = value
    return merged


def register(args: argparse.Namespace) -> None:
    mode = args.benchmark_mode
    validate_mode(mode)
    reg_path = registry_path(mode)
    if not reg_path.is_file():
        raise RuntimeError(f"gallery is not initialized: {reg_path}; run scripts/initialize_eval_gallery.sh first")
    results_dir, evaluation_dir = args.results_dir.resolve(), args.evaluation_dir.resolve()
    summary_path, mapping_path = evaluation_dir / "summary.json", evaluation_dir / "mapping.json"
    for path in (results_dir, summary_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(path)
    with registry_lock(mode):
        registry = load_json(reg_path)
        manifest_path = Path(registry["manifest"])
        if sha256(manifest_path) != registry["manifest_sha256"]:
            raise RuntimeError(f"registered {mode} manifest changed: {manifest_path}")
        manifest = load_json(manifest_path)
        expected = [str(row["case_id"]) for row in manifest["cases"]]
        mapping = load_json(mapping_path)
        actual = [str(row.get("case_id", row.get("name"))) for row in mapping]
        if actual != expected:
            raise RuntimeError(f"{mode} registration requires the complete ordered manifest: expected {len(expected)} cases, got {len(actual)}")
        prefix_ok = all(("-long-" in case_id) == (mode == "long") for case_id in actual)
        if not prefix_ok:
            raise RuntimeError(f"mapping case IDs do not match benchmark mode {mode}")
        summary = load_json(summary_path)
        if int(summary.get("case_count") or 0) != len(expected):
            raise RuntimeError(f"summary case_count mismatch: {summary.get('case_count')} != {len(expected)}")
        missing = [row["case_id"] for row in mapping if not Path(row["generated"]).is_file()]
        if missing:
            raise RuntimeError(f"missing generated videos, first entries: {missing[:5]}")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        run_id = safe_id(args.run_id or results_dir.name)
        failures = summary.get("failure_count", {})
        status = "partial" if any(int(value or 0) > 0 for value in failures.values()) else "complete"
        entry = {
            "run_id": run_id, "label": args.label or run_id, "group": args.group or "Other",
            "tags": [item.strip() for item in args.tags.split(",") if item.strip()],
            "benchmark_mode": mode, "model_path": args.model_path or "",
            "model_url": args.model_url or "", "checkpoint_step": args.checkpoint_step,
            "inference_steps": args.inference_steps, "seed": args.seed,
            "results_dir": results_dir.as_posix(), "evaluation_dir": evaluation_dir.as_posix(),
            "summary_path": summary_path.as_posix(), "mapping_path": mapping_path.as_posix(),
            "manifest_sha256": registry["manifest_sha256"], "status": status,
            "notes": args.notes or "", "updated_at": now,
        }
        old = next((row for row in registry["runs"] if row["run_id"] == run_id), None)
        if old:
            entry["created_at"] = old.get("created_at", now)
            registry["runs"] = [entry if row["run_id"] == run_id else row for row in registry["runs"]]
        else:
            entry["created_at"] = now
            registry["runs"].append(entry)
        write_json_atomic(reg_path, registry)
    build(mode)
    print(f"[gallery-register] {mode}/{run_id} -> {reg_path}")


def build(mode: str) -> None:
    validate_mode(mode)
    reg_path = registry_path(mode)
    if not reg_path.is_file():
        raise FileNotFoundError(reg_path)
    registry = load_json(reg_path)
    manifest_path = Path(registry["manifest"])
    manifest = load_json(manifest_path)
    report_dir = REPORT_ROOT / mode
    media_dir = report_dir / "media"
    report_dir.mkdir(parents=True, exist_ok=True)
    ensure_link(media_dir / "benchmark", manifest_path.parent)
    cases = []
    for row in manifest["cases"]:
        item = dict(row)
        for key in ("ref_image", "origin_video", "origin_mask", "gan_swapped_video"):
            if item.get(key):
                item[key] = f"media/benchmark/{item[key]}"
        cases.append(item)
    runs = []
    for entry in registry.get("runs", []):
        summary = load_json(Path(entry["summary_path"]))
        mapping = load_json(Path(entry["mapping_path"]))
        run_id = safe_id(entry["run_id"])
        videos: dict[str, str] = {}
        for row in mapping:
            case_id = str(row.get("case_id", row.get("name")))
            target = Path(row["generated"])
            link = media_dir / "runs" / run_id / f"{case_id}{target.suffix.lower()}"
            ensure_link(link, target)
            videos[case_id] = f"media/runs/{run_id}/{link.name}"
        runs.append({**entry, "summary": summary, "metrics": {
            key: metric_number(value) for key, value in summary.get("metrics_flat", {}).items()
        }, "case_metrics": case_metrics(Path(entry["evaluation_dir"]), summary), "videos": videos})
    payload = {
        "schema_version": 1, "benchmark_mode": mode, "manifest": manifest_path.as_posix(),
        "manifest_sha256": registry["manifest_sha256"], "cases": cases, "runs": runs,
        "metrics": [{"key": key, "label": spec[0], "higher": spec[1], "digits": spec[2]}
                    for key, spec in METRICS.items()],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json_atomic(report_dir / "data.json", payload)
    (report_dir / "index.html").write_text(page_html(mode), encoding="utf-8")
    print(f"[gallery-build] {mode}: {len(runs)} runs, {len(cases)} cases -> {report_dir / 'index.html'}")


def page_html(mode: str) -> str:
    title = f"SwapFace {mode.capitalize()} Benchmark"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#f4f6f8;--panel:#fff;--line:#d9dee7;--ink:#172033;--muted:#667085;--blue:#1769c2;--best:#087443;--bestbg:#e9f8ef;--second:#8a5b00;--secondbg:#fff6d9}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:13px Arial,sans-serif}}header{{position:sticky;top:0;z-index:5;padding:13px 18px;background:#fffffff3;border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}h1{{font-size:19px;margin:0}}.meta{{color:var(--muted);margin-top:5px}}main{{max-width:1900px;margin:auto;padding:16px}}.tools{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}}input,select,button{{height:32px;border:1px solid #c7ced9;border-radius:6px;background:#fff;padding:0 9px;color:var(--ink)}}button{{cursor:pointer}}.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:7px}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:7px 8px;border-bottom:1px solid #e8ebef;text-align:left;font-size:11px}}th{{background:#eef2f6;position:sticky;top:0}}tr.hidden{{display:none}}.best{{background:var(--bestbg);color:var(--best);font-weight:700}}.second{{background:var(--secondbg);color:var(--second);font-weight:700}}.status{{border-radius:10px;padding:2px 7px;background:#e8f5ed;color:#087443}}section{{margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:7px;padding:12px}}.casebar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.casebar strong{{margin-right:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin-top:10px}}.card{{border:1px solid var(--line);border-radius:6px;overflow:hidden}}.head{{padding:7px 9px;font-weight:700}}video,img,.missing{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#111}}.missing{{display:grid;place-items:center;color:#bbb}}.small{{padding:7px 9px;color:var(--muted);font-size:11px;overflow-wrap:anywhere}}details{{margin-top:5px}}@media(max-width:700px){{main{{padding:10px}}header{{position:static}}}}
</style></head><body><header><h1>{html.escape(title)}</h1><div class="meta" id="meta">加载中…</div></header><main>
<div class="tools"><input id="search" placeholder="搜索实验、分组或标签"><select id="sort"><option value="group">按分组</option></select><button id="showAll">全部显示</button><button id="hideAll">全部隐藏</button></div>
<div class="table-wrap"><table><thead><tr id="head"><th>显示</th><th>实验</th><th>组</th><th>Steps</th><th>Seed</th><th>状态</th></tr></thead><tbody id="runs"></tbody></table></div>
<section><div class="casebar"><strong id="caseTitle">Case</strong><button id="prev">上一个</button><select id="caseSelect"></select><button id="next">下一个</button><button id="play">全部播放</button><button id="pause">暂停</button><button id="restart">归零</button></div><div class="grid" id="grid"></div></section>
</main><script>
let DATA,state={{visible:{{}},caseIndex:0,query:'',sort:'group'}};const storage='swapface-gallery::{mode}';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function loadState(){{try{{Object.assign(state,JSON.parse(localStorage.getItem(storage)||'{{}}'))}}catch(e){{}}}}function save(){{localStorage.setItem(storage,JSON.stringify(state))}}
function value(run,key){{const v=run.metrics[key];return Number.isFinite(v)?v:null}}function fmt(v,m){{return v===null?'-':v.toFixed(m.digits)}}
function rankings(key){{const m=DATA.metrics.find(x=>x.key===key),xs=DATA.runs.map(r=>[r.run_id,value(r,key)]).filter(x=>x[1]!==null).sort((a,b)=>m.higher?b[1]-a[1]:a[1]-b[1]);return new Map(xs.map((x,i)=>[x[0],i+1]))}}
function renderTable(){{const metrics=DATA.metrics.filter(m=>DATA.runs.some(r=>value(r,m.key)!==null));head.innerHTML='<th>显示</th><th>实验</th><th>组</th><th>Steps</th><th>Seed</th><th>状态</th>'+metrics.map(m=>`<th title="${{esc(m.label)}}">${{esc(m.label)}} ${{m.higher?'↑':'↓'}}</th>`).join('');sort.innerHTML='<option value="group">按分组</option>'+metrics.map(m=>`<option value="${{m.key}}">按 ${{esc(m.label)}}</option>`).join('');sort.value=state.sort;const ranks=Object.fromEntries(metrics.map(m=>[m.key,rankings(m.key)]));let rs=[...DATA.runs];if(state.sort!=='group'){{const m=metrics.find(x=>x.key===state.sort);rs.sort((a,b)=>{{const x=value(a,m.key),y=value(b,m.key);if(x===null)return 1;if(y===null)return-1;return m.higher?y-x:x-y}})}}else rs.sort((a,b)=>(a.group||'').localeCompare(b.group||'')||a.label.localeCompare(b.label));runs.innerHTML=rs.map(r=>{{if(state.visible[r.run_id]===undefined)state.visible[r.run_id]=true;const hay=[r.label,r.group,...r.tags].join(' ').toLowerCase(),filtered=state.query&&!hay.includes(state.query.toLowerCase());return `<tr data-id="${{esc(r.run_id)}}" class="${{filtered?'hidden':''}}"><td><input class="toggle" type="checkbox" ${{state.visible[r.run_id]?'checked':''}}></td><td><b>${{esc(r.label)}}</b><details><summary>详情</summary><div>${{esc(r.model_path||r.model_url||'')}}<br>${{esc(r.notes||'')}}<br>${{esc(r.evaluation_dir)}}</div></details></td><td>${{esc(r.group)}}</td><td>${{r.inference_steps??'-'}}</td><td>${{r.seed??'-'}}</td><td><span class="status">${{esc(r.status)}}</span></td>`+metrics.map(m=>{{const rank=ranks[m.key].get(r.run_id),cls=rank===1?'best':rank===2?'second':'';return `<td class="${{cls}}">${{fmt(value(r,m.key),m)}}${{rank?' · #'+rank:''}}</td>`}}).join('')+'</tr>'}}).join('');document.querySelectorAll('.toggle').forEach(x=>x.onchange=()=>{{state.visible[x.closest('tr').dataset.id]=x.checked;save();renderCase()}});save()}}
function media(label,src,type='video',note=''){{return `<article class="card"><div class="head">${{esc(label)}}</div>${{src?(type==='image'?`<img loading="lazy" src="${{esc(src)}}">`:`<video controls muted loop playsinline preload="metadata" src="${{esc(src)}}"></video>`):'<div class="missing">Missing</div>'}}<div class="small">${{note}}</div></article>`}}
function renderCase(){{const c=DATA.cases[state.caseIndex];caseTitle.textContent=c.case_id;caseSelect.value=String(state.caseIndex);let cards=media('Reference image',c.ref_image,'image',esc(c.source_identity||''))+media('Origin video',c.origin_video,'video',esc(`${{c.fps||''}} fps`))+media('GAN swapped',c.gan_swapped_video,'video','benchmark baseline');for(const r of DATA.runs)if(state.visible[r.run_id]!==false){{const cm=r.case_metrics[c.case_id]||{{}},note=Object.entries(cm).map(([k,v])=>`${{esc((DATA.metrics.find(m=>m.key===k)||{{label:k}}).label)}} <b>${{Number(v).toFixed((DATA.metrics.find(m=>m.key===k)||{{digits:4}}).digits)}}</b>`).join(' · ')||`steps=${{r.inference_steps??'-'}} · seed=${{r.seed??'-'}}`;cards+=media(r.label,r.videos[c.case_id],'video',note)}}grid.innerHTML=cards;save()}}
fetch('data.json').then(r=>r.json()).then(d=>{{DATA=d;loadState();state.caseIndex=Math.max(0,Math.min(state.caseIndex,d.cases.length-1));meta.textContent=`${{d.cases.length}} cases · ${{d.runs.length}} registered evals · updated ${{d.generated_at}}`;caseSelect.innerHTML=d.cases.map((c,i)=>`<option value="${{i}}">${{c.case_id}}</option>`).join('');renderTable();renderCase()}}).catch(e=>meta.textContent='加载失败: '+e);
search.oninput=()=>{{state.query=search.value;renderTable()}};sort.onchange=()=>{{state.sort=sort.value;renderTable()}};showAll.onclick=()=>{{DATA.runs.forEach(r=>state.visible[r.run_id]=true);renderTable();renderCase()}};hideAll.onclick=()=>{{DATA.runs.forEach(r=>state.visible[r.run_id]=false);renderTable();renderCase()}};caseSelect.onchange=()=>{{state.caseIndex=+caseSelect.value;renderCase()}};prev.onclick=()=>{{state.caseIndex=(state.caseIndex-1+DATA.cases.length)%DATA.cases.length;renderCase()}};next.onclick=()=>{{state.caseIndex=(state.caseIndex+1)%DATA.cases.length;renderCase()}};play.onclick=()=>document.querySelectorAll('video').forEach(v=>v.play().catch(()=>{{}}));pause.onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());restart.onclick=()=>document.querySelectorAll('video').forEach(v=>{{v.pause();v.currentTime=0}});
</script></body></html>'''


def initialize(args: argparse.Namespace) -> None:
    for mode, manifest in (("short", args.short_manifest), ("long", args.long_manifest)):
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        path = registry_path(mode)
        with registry_lock(mode):
            if path.exists() and not args.force:
                current = load_json(path)
                if Path(current["manifest"]).resolve() != manifest.resolve():
                    raise RuntimeError(f"{path} already uses a different manifest; pass --force to reset")
            else:
                write_json_atomic(path, new_registry(mode, manifest))
        build(mode)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--short-manifest", type=Path, required=True)
    init.add_argument("--long-manifest", type=Path, required=True)
    init.add_argument("--force", action="store_true")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--benchmark-mode", choices=("short", "long", "all"), default="all")
    reg = sub.add_parser("register")
    reg.add_argument("--benchmark-mode", choices=("short", "long"), required=True)
    reg.add_argument("--results-dir", type=Path, required=True)
    reg.add_argument("--evaluation-dir", type=Path, required=True)
    reg.add_argument("--run-id", default="")
    reg.add_argument("--label", default="")
    reg.add_argument("--group", default="Other")
    reg.add_argument("--tags", default="")
    reg.add_argument("--model-path", default="")
    reg.add_argument("--model-url", default="")
    reg.add_argument("--checkpoint-step", type=int)
    reg.add_argument("--inference-steps", type=int)
    reg.add_argument("--seed", type=int)
    reg.add_argument("--notes", default="")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "init": initialize(args)
    elif args.command == "build":
        for mode in (("short", "long") if args.benchmark_mode == "all" else (args.benchmark_mode,)): build(mode)
    else: register(args)


if __name__ == "__main__":
    main()
