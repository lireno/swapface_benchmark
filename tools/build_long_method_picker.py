#!/usr/bin/env python3
"""Build an isolated 200-case method picker using the existing gallery template."""
import json
from pathlib import Path
from datetime import datetime, timezone
from eval_gallery import page_html, ensure_link
from concurrent.futures import ThreadPoolExecutor
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'web_reports/long_method_picker'
BENCH = Path('/mnt/cpfs/users/lzk/dataset/swapface_benchmark/long_200/benchmark')
LIVING = Path('/mnt/cpfs/users/lzk/codes/swap_face/HiFiVFS_wan/outputs/livingswap_swapface_benchmark/full_4gpu/long/results')
OURS = Path('/mnt/cpfs/users/lzk/codes/swap_face/idvtrain/output/lzk_rollout_distillation/evaluation/model_eval/rollout3p9_continue500opt_long200_h3c9/configs/7_h3c9/generation/videos')

def main():
    cases = json.loads((BENCH / 'manifest.json').read_text())['cases']
    runs = []
    for name, label, folder in [('livingswap', 'LivingSwap', LIVING), ('ours', 'Ours · rollout 3+9', OURS)]:
        videos = {}
        for c in cases:
            matches = sorted(folder.glob(c['case_id'] + '*.mp4'))
            if len(matches) != 1:
                raise RuntimeError(f'{name} {c["case_id"]}: expected one video, got {matches}')
            link = OUT / 'media' / name / (c['case_id'] + '.mp4')
            ensure_link(link, matches[0])
            videos[c['case_id']] = str(link.relative_to(OUT))
        runs.append(dict(run_id=name, label=label, group='Methods', tags=[], status='200 videos',
                         metrics={}, case_metrics={}, videos=videos, evaluation_dir=str(folder)))
    ensure_link(OUT / 'media/benchmark', BENCH)
    for c in cases:
        for key in ('ref_image', 'origin_video', 'gan_swapped_video'):
            if not (BENCH / c[key]).is_file():
                raise FileNotFoundError(BENCH / c[key])
            c[key] = 'media/benchmark/' + c[key]
    # Read each file's own FPS rather than assuming all methods encode at 24 FPS.
    urls = [c[k] for c in cases for k in ('origin_video', 'gan_swapped_video')]
    urls += [url for run in runs for url in run['videos'].values()]
    def probe(url):
        cap = cv2.VideoCapture(str(OUT / url))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or fps <= 0 or count <= 0:
                raise RuntimeError(f'Cannot read frame metadata: {url}')
            return url, dict(fps=fps, frames=count)
        finally:
            cap.release()
    with ThreadPoolExecutor(max_workers=4) as pool:
        video_metadata = dict(pool.map(probe, urls))
    payload = dict(cases=cases, runs=runs, metrics=[], video_metadata=video_metadata,
                   generated_at=datetime.now(timezone.utc).isoformat())
    (OUT / 'data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    page = page_html('long')
    page = page.replace('SwapFace Long Benchmark', 'Long 200 · 方法对比与 Case 筛选')
    page = page.replace('swapface-gallery::long', 'swapface-gallery::long-method-picker')
    page = page.replace("media('GAN swapped'", "media('Inswapper'")
    page = page.replace("media('Reference image'", "media('Target identity image'")
    page = page.replace(' registered evals', ' methods + Inswapper')
    # Explicitly cancel old media requests before replacing a case. Detached
    # videos otherwise continue buffering and can exhaust browser connections.
    page = page.replace('function renderCase(){', '''function renderCase(){
document.querySelectorAll('#grid video').forEach(v=>{v.pause();v.removeAttribute('src');v.load();});''')
    page = page.replace('</main><script>', '''<section><div class="tools">
<button id="pick">收藏当前 case</button><button id="exportPicks">导出选中 case JSON</button>
<span id="picked"></span></div><div class="tools"><label>同步时间（秒） <input id="seekAll" type="number" min="0" step="0.04" value="0"></label>
<button id="seekButton">所有视频跳转</button></div><div id="pickList"></div></section></main><script>''')
    page = page.replace('</body>', '''<script>
const pickKey='swapface-long-method-picker-favorites';
let picks=JSON.parse(localStorage.getItem(pickKey)||'[]');
function renderPicks(){document.getElementById('picked').textContent=`已收藏 ${picks.length} 个 case（保存在本浏览器）`;
document.getElementById('pickList').textContent=picks.join(', ');}
document.getElementById('pick').onclick=()=>{if(!DATA)return;const id=DATA.cases[state.caseIndex].case_id;
picks=picks.includes(id)?picks.filter(x=>x!==id):[...picks,id];localStorage.setItem(pickKey,JSON.stringify(picks));renderPicks();};
document.getElementById('exportPicks').onclick=()=>{const b=new Blob([JSON.stringify({case_ids:picks,methods:['Inswapper','LivingSwap','Ours rollout3p9_continue500opt']},null,2)],{type:'application/json'});
const url=URL.createObjectURL(b),a=document.createElement('a');a.href=url;a.download='selected_long_cases.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.getElementById('seekButton').onclick=()=>{let t=Number(document.getElementById('seekAll').value);if(!Number.isFinite(t)||t<0)return;
document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=Number.isFinite(v.duration)?Math.min(t,v.duration):t;});};
renderPicks();
</script></body>''')
    page = page.replace('</body>', '''<script>
// Frame indices are zero based, matching extraction scripts. These benchmark
// MP4s use constant frame rates; mediaTime identifies the displayed frame.
function attachFrameControls(v){
  if(v.dataset.frameControls)return;
  v.dataset.frameControls='1';
  const m=DATA.video_metadata[v.getAttribute('src')];
  if(!m)return;
  const box=document.createElement('div');box.className='small';
  box.innerHTML='<div class="frame-status"></div><div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:6px"><button data-delta="-1">上一帧</button><button data-delta="1">下一帧</button><input type="number" min="0" step="1" style="width:90px" aria-label="帧索引"><button class="jump">跳帧</button><button class="sync">同步其他视频</button></div>';
  v.after(box);
  const label=box.querySelector('.frame-status'),input=box.querySelector('input');
  input.max=m.frames-1;input.value=0;
  let frame=0;
  function update(time){
    frame=Math.max(0,Math.min(m.frames-1,Math.floor(time*m.fps+1e-4)));
    label.textContent=`第 ${frame+1} / ${m.frames} 帧 · 索引 ${frame}（从0开始） · ${time.toFixed(3)}s · ${m.fps.toFixed(3)} FPS`;
    if(document.activeElement!==input)input.value=frame;
  }
  function jump(n){
    n=Math.max(0,Math.min(m.frames-1,Math.round(n)));
    if(!Number.isFinite(n))return;
    queueVideoSeek(v,(n+0.1)/m.fps);
  }
  box.querySelectorAll('[data-delta]').forEach(b=>b.onclick=()=>jump(frame+Number(b.dataset.delta)));
  box.querySelector('.jump').onclick=()=>jump(Number(input.value));
  input.onkeydown=e=>{if(e.key==='Enter')jump(Number(input.value));};
  box.querySelector('.sync').onclick=()=>{
    const t=(frame+0.1)/m.fps;
    masterTarget=t;masterSeek.value=t;masterVideos().forEach(other=>queueVideoSeek(other,t));
  };
  v.addEventListener('loadedmetadata',()=>update(v.currentTime));
  v.addEventListener('seeked',()=>update(v.currentTime));
  if(v.requestVideoFrameCallback){
    const cb=(_,meta)=>{update(meta.mediaTime);if(v.isConnected)v.requestVideoFrameCallback(cb);};
    v.requestVideoFrameCallback(cb);
  }else v.addEventListener('timeupdate',()=>update(v.currentTime));
  update(0);
}
new MutationObserver(()=>document.querySelectorAll('video').forEach(attachFrameControls))
  .observe(document.getElementById('grid'),{childList:true});
</script></body>''')
    page = page.replace('<div class="grid" id="grid">', '''<div style="display:flex;align-items:center;gap:10px;margin-top:12px">
<label for="masterSeek">总进度</label><input id="masterSeek" type="range" min="0" max="0" step="0.001" value="0" disabled style="flex:1;padding:0" aria-label="所有视频的总进度">
<span id="masterTime">0.000 / 0.000 s</span></div><div class="grid" id="grid">''')
    page = page.replace('</body>', '''<script>
const masterSeek=document.getElementById('masterSeek'),masterTime=document.getElementById('masterTime');
let masterDragging=false,masterTarget=null,seekTimer=null;
function queueVideoSeek(v,time){
  v.pause();v._pendingTime=time;
  if(v.readyState<1 || !Number.isFinite(v.duration) || v.seeking)return;
  const target=Math.max(0,Math.min(time,v.duration-0.001));
  v._pendingTime=null;
  try{v.currentTime=target;}catch(e){v._pendingTime=time;}
}
function masterVideos(){return [...document.querySelectorAll('#grid video')];}
function masterDuration(){
  const videos=masterVideos();
  // Common duration prevents shorter methods from ending before the others.
  const durations=videos.map(v=>{const m=DATA.video_metadata[v.getAttribute('src')];return m?m.frames/m.fps:v.duration;});
  return durations.length&&durations.every(d=>Number.isFinite(d)&&d>0)?Math.min(...durations):0;
}
function updateMaster(){
  const duration=masterDuration(),v=masterVideos()[0];
  masterSeek.disabled=!duration;masterSeek.max=duration;
  if(masterTarget!==null && masterVideos().every(x=>x.readyState>=2 && !x.seeking && x._pendingTime==null && Math.abs(x.currentTime-masterTarget)<0.08))masterTarget=null;
  if(!masterDragging && masterTarget===null)masterSeek.value=Math.min(v?.currentTime||0,duration);
  masterTime.textContent=`${Number(masterSeek.value).toFixed(3)} / ${duration.toFixed(3)} s`;
}
masterSeek.addEventListener('pointerdown',()=>{masterDragging=true;});
function seekMaster(){
  const time=Number(masterSeek.value);
  masterTarget=time;
  clearTimeout(seekTimer);
  masterVideos().forEach(v=>v.pause());
  // Coalesce drag events: do not repeatedly cancel network range requests.
  seekTimer=setTimeout(()=>masterVideos().forEach(v=>queueVideoSeek(v,time)),120);
  masterTime.textContent=`${time.toFixed(3)} / ${masterDuration().toFixed(3)} s`;
}
masterSeek.addEventListener('input',seekMaster);
function finishMasterDrag(){if(!masterDragging)return;masterDragging=false;}
masterSeek.addEventListener('change',finishMasterDrag);
window.addEventListener('pointerup',finishMasterDrag);
window.addEventListener('pointercancel',finishMasterDrag);
new MutationObserver(()=>{
  clearTimeout(seekTimer);
  masterDragging=false;masterTarget=null;masterSeek.value=0;
  masterVideos().forEach(v=>{
    v.preload='auto';
    const status=document.createElement('div');status.className='small';status.textContent='视频加载中…';v.after(status);
    const ready=()=>{status.textContent='已就绪';if(v._pendingTime!=null)queueVideoSeek(v,v._pendingTime);updateMaster();};
    v.addEventListener('loadedmetadata',ready);
    v.addEventListener('canplay',ready);
    v.addEventListener('durationchange',updateMaster);
    v.addEventListener('timeupdate',updateMaster);
    v.addEventListener('seeked',ready);
    v.addEventListener('seeking',()=>{status.textContent='定位中…';});
    v.addEventListener('error',()=>{status.textContent=`加载失败，错误码 ${v.error?.code}；请确认已使用新版启动脚本`;});
    v.load();
  });
  updateMaster();
}).observe(document.getElementById('grid'),{childList:true});
document.getElementById('play').onclick=()=>masterVideos().forEach(v=>{
  v.play().catch(e=>{const card=v.closest('.card');card.querySelector('.head').title=`播放失败: ${e.message}`;});
});
document.getElementById('restart').onclick=()=>{masterTarget=0;masterSeek.value=0;masterVideos().forEach(v=>queueVideoSeek(v,0));};
document.getElementById('seekButton').onclick=()=>{
 const t=Number(document.getElementById('seekAll').value);if(!Number.isFinite(t)||t<0)return;
 masterTarget=Math.min(t,masterDuration());masterSeek.value=masterTarget;masterVideos().forEach(v=>queueVideoSeek(v,masterTarget));
};
</script></body>''')
    (OUT / 'index.html').write_text(page)
    print(f'Built {len(cases)} cases, 3 methods: {OUT}')

if __name__ == '__main__':
    main()
