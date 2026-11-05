"""Live-view web server (``camthink_edge liveview --root ROOT``): one reusable page for any
object-detection project.

It is a separate process from the inference service and only *reads* what the monitor
publishes in the shared-memory directory (``stats.json``, ``chN.jpg``/``chN.json``), plus the
heartbeat, the current deployment's ``runtime.json`` and ``app.json`` (credentials masked).
It writes a single marker, ``viewers.json``, so the monitor encodes frames only while someone
is watching. Stopping this server never affects inference; stopping inference leaves this page
up, showing why there is no video.

Endpoints (stdlib ``ThreadingHTTPServer``; no framework needed on the device):

    /                     the page
    /api/status           everything the page shows (model, device, inference, metrics, config)
    /stats.json           alias of /api/status
    /api/detections?ch=N  the last inferred frame's detections (metadata only)
    /stream.mjpg?ch=N     MJPEG with boxes (``fps`` and ``q`` query parameters optional)
    /snapshot.jpg?ch=N    one annotated JPEG
    /healthz              200 {"ok": true} when the server runs
"""

from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import VERSION, appcfg, overlay

STATS_FRESH_S = 5.0
HEARTBEAT_FRESH_S = 15.0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


class LiveState:
    def __init__(self, root: Path, *, port: int):
        self.root = Path(root)
        self.live = appcfg.live_dir(self.root)
        self.port = port
        self.started = time.time()
        self._lock = threading.Lock()
        self.viewers = 0
        self.streams_served = 0
        self.frames_sent = 0
        self.bytes_sent = 0
        self._placeholder_cache: dict[str, bytes] = {}

    # -- viewer bookkeeping --------------------------------------------------------------------

    def viewer_enter(self) -> None:
        with self._lock:
            self.viewers += 1
            self.streams_served += 1
        self.write_viewers()

    def viewer_leave(self) -> None:
        with self._lock:
            self.viewers = max(self.viewers - 1, 0)
        self.write_viewers()

    def write_viewers(self, *, force_count: int | None = None) -> None:
        try:
            self.live.mkdir(parents=True, exist_ok=True)
            tmp = self.live / "viewers.json.tmp"
            tmp.write_text(
                json.dumps(
                    {
                        "count": self.viewers if force_count is None else force_count,
                        "ts": time.time(),
                        "pid": os.getpid(),
                    }
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self.live / "viewers.json")
        except OSError:
            pass

    # -- data -----------------------------------------------------------------------------------

    def current(self) -> str | None:
        try:
            return os.path.basename(os.readlink(self.root / "current"))
        except OSError:
            return None

    def config(self) -> dict[str, Any]:
        try:
            return appcfg.load(self.root)
        except Exception:  # noqa: BLE001
            return appcfg.merge_defaults(None)

    def model_info(self, current: str | None) -> dict[str, Any] | None:
        if not current:
            return None
        dep = self.root / "deployments" / current
        rt = _read_json(dep / "runtime.json") or {}
        meta = _read_json(dep / "deployment.json") or {}
        eng = _read_json(dep / "model.engine.json") or {}
        post = rt.get("postprocess") or {}
        project, _, dep_id = current.rpartition("-dep-")
        return {
            "target_ref": current,
            "project": rt.get("project") or meta.get("project") or project or None,
            "deployment_id": meta.get("deployment_id") or (f"dep-{dep_id}" if dep_id else None),
            "model_version": rt.get("model_version") or meta.get("model_version"),
            "export_id": rt.get("export_id") or meta.get("export_id"),
            "classes": rt.get("classes") or [],
            "class_colours": [
                overlay.class_colour_hex(i) for i in range(len(rt.get("classes") or []))
            ],
            "input_size": rt.get("input_size"),
            "precision": rt.get("precision") or eng.get("precision"),
            "confidence_default": post.get("confidence"),
            "iou": post.get("iou"),
            "tensorrt": eng.get("tensorrt"),
        }

    def status(self) -> dict[str, Any]:
        now = time.time()
        cfg = self.config()
        current = self.current()
        hb = _read_json(self.root / "service" / "status.json")
        stopped = _read_json(self.root / "service" / "stopped.json")
        stats = _read_json(self.live / "stats.json")
        stats_age = (now - float(stats.get("ts_epoch", 0))) if stats else None
        monitor_fresh = (
            stats is not None and stats.get("state") == "running" and stats_age <= STATS_FRESH_S
        )
        hb_age = (now - float(hb.get("ts_epoch", 0))) if hb else None
        if stopped:
            inf_state, inf_reason = "stopped", "stopped by operator (service stop)"
        elif hb is None:
            inf_state, inf_reason = "unknown", "no heartbeat file"
        elif hb.get("error"):
            inf_state, inf_reason = "failed", str(hb["error"])
        elif hb.get("state") == "stopped":
            inf_state, inf_reason = "stopped", "inference service exited"
        elif hb_age is not None and hb_age > HEARTBEAT_FRESH_S:
            inf_state, inf_reason = "stale", f"heartbeat {hb_age:.0f}s old"
        elif hb.get("mode") != "rtsp":
            inf_state, inf_reason = "running", "inference runs on sample images (no camera)"
        else:
            inf_state, inf_reason = "running", None
        model = self.model_info(current)
        if monitor_fresh and stats.get("model"):
            model = {
                **(model or {}),
                **{
                    k: v
                    for k, v in stats["model"].items()
                    if k in ("confidence", "workers", "channels", "decode_backend", "engine")
                },
            }
        elif model is not None:
            model["confidence"] = (
                cfg["inference"].get("confidence")
                if cfg["inference"].get("confidence") is not None
                else model.get("confidence_default")
            )
        chans = stats.get("channels", []) if monitor_fresh else []
        return {
            "ts": now,
            "liveview": {
                "runtime_version": VERSION,
                "port": self.port,
                "uptime_s": round(now - self.started, 1),
                "viewers": self.viewers,
                "streams_served": self.streams_served,
                "frames_sent": self.frames_sent,
                "bytes_sent": self.bytes_sent,
                "process_rss_mb": _rss_mb(),
                "display_fps": cfg["liveview"].get("display_fps"),
                "quality": cfg["liveview"].get("quality"),
            },
            "device": {
                "hostname": socket.gethostname(),
                "name": (cfg.get("device") or {}).get("name"),
                "label": (cfg.get("device") or {}).get("label"),
                "power_mode": stats.get("power_mode") if stats else None,
            },
            "deployment": current,
            "model": model,
            "inference": {
                "state": inf_state,
                "reason": inf_reason,
                "mode": (hb or {}).get("mode", "samples" if hb else None),
                "heartbeat_age_s": round(hb_age, 1) if hb_age is not None else None,
                "pid": (hb or {}).get("pid"),
                "deployment": (hb or {}).get("deployment"),
                "frames": (hb or {}).get("frames"),
                "errors": (hb or {}).get("errors"),
                "stopped": stopped,
                "config_restart_required": (hb or {}).get("config_restart_required", False),
            },
            "monitor": (
                {
                    "fresh": monitor_fresh,
                    "stats_age_s": round(stats_age, 1) if stats_age is not None else None,
                    "state": stats.get("state"),
                    "uptime_s": stats.get("uptime_s"),
                    "process_rss_mb": stats.get("process_rss_mb"),
                    "camera_connected": stats.get("camera_connected"),
                    "live": stats.get("live"),
                    "config_restart_required": stats.get("config_restart_required"),
                }
                if stats
                else None
            ),
            "channels": chans,
            "system": (stats.get("system") if monitor_fresh else None),
            "system_backend": stats.get("system_backend") if stats else None,
            "config": {
                "sources": [
                    {"name": n, "url": appcfg.mask_url(s.get("url"))}
                    for n, s in zip(  # noqa: B905 - py3.8 runtime
                        appcfg.source_names(cfg), cfg.get("sources", [])
                    )
                ],
                "inference": cfg["inference"],
                "liveview": cfg["liveview"],
            },
        }

    # -- frames ---------------------------------------------------------------------------------

    def frame(self, ch: int) -> tuple[bytes | None, dict[str, Any] | None]:
        """(JPEG bytes or None, metadata or None). The metadata's ``seq``/``ts`` describe the
        newest inference; ``jpg_seq``/``jpg_ts`` the JPEG (absent until someone watches)."""
        meta = _read_json(self.live / f"ch{ch}.json")
        try:
            data = (self.live / f"ch{ch}.jpg").read_bytes()
        except OSError:
            data = None
        if meta is not None and meta.get("jpg_seq") is None:
            data = None
        return data, meta

    @staticmethod
    def jpg_fresh(meta: dict[str, Any] | None) -> bool:
        return bool(meta) and time.time() - float(meta.get("jpg_ts") or 0) < STATS_FRESH_S

    def placeholder(self, text: str, size: tuple[int, int] = (960, 540)) -> bytes:
        key = f"{text}|{size}"
        if key not in self._placeholder_cache:
            img = overlay.placeholder(size[0], size[1], text)
            self._placeholder_cache[key] = overlay.encode_jpeg(img, 60) or b""
        return self._placeholder_cache[key]

    def no_video_reason(self, ch: int) -> str:
        st = self.status()
        inf = st["inference"]
        if inf["state"] != "running":
            return f"inference {inf['state']}: {inf['reason']}"
        if inf.get("mode") != "rtsp":
            return "no camera configured (app source set)"
        if not st["monitor"] or not st["monitor"]["fresh"]:
            return "waiting for the inference service"
        chans = st["channels"]
        if ch >= len(chans):
            return f"channel {ch} does not exist"
        src = chans[ch]["source"]
        if not src["connected"]:
            return f"camera disconnected: {src.get('error') or 'reconnecting'}"
        return "waiting for the first inferred frame"


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CamThink live view</title>
<style>
:root{--bg:#0f1317;--card:#171d24;--line:#26303a;--fg:#dfe6ee;--mute:#8fa1b3;--ok:#1f8a4c;--bad:#b23a3a;--warn:#b07a12}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0 8px 0 0;font-weight:600}
.badge{padding:3px 9px;border-radius:12px;background:#263240;font-size:12px;white-space:nowrap}
.ok{background:var(--ok)}.bad{background:var(--bad)}.warn{background:var(--warn)}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:12px;padding:12px}
@media(max-width:1000px){main{grid-template-columns:1fr}}
#video{width:100%;max-height:78vh;object-fit:contain;background:#000;border-radius:6px;display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin-bottom:10px}
.card h2{font-size:12px;margin:0 0 8px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse}td{padding:2px 0;vertical-align:top}
td:first-child{color:var(--mute)}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 5px 0 0;vertical-align:-1px}
.cls{display:inline-block;margin:0 10px 4px 0}
select{background:#263240;color:var(--fg);border:1px solid #38454f;border-radius:4px;padding:3px 6px}
canvas{width:100%;height:46px;display:block;background:#10151a;border-radius:4px;margin:4px 0 8px}
.lbl{display:flex;justify-content:space-between;color:var(--mute);font-size:12px}
small{color:var(--mute)}
</style></head><body>
<header><h1 id="title">CamThink live view</h1>
<span id="chsel" style="display:none">channel <select id="ch"></select></span>
<span id="b-cam" class="badge">camera …</span><span id="b-inf" class="badge">inference …</span>
<span id="b-dev" class="badge"></span><span id="b-model" class="badge"></span><span id="b-view" class="badge"></span>
</header>
<main>
<div>
<img id="video" src="/stream.mjpg?ch=0" alt="live video">
<div class="card" style="margin-top:10px"><h2>Detections in the current frame</h2><div id="dets"><small>–</small></div>
<div id="legend" style="margin-top:6px"></div></div>
</div>
<div>
<div class="card"><h2>Model</h2><table id="mdl"></table></div>
<div class="card"><h2>Performance</h2><table id="perf"></table></div>
<div class="card"><h2>Trends (last 2 min)</h2>
<div class="lbl"><span>detection fps</span><span id="v-fps"></span></div><canvas id="c-fps"></canvas>
<div class="lbl"><span>inference latency p50 (ms)</span><span id="v-lat"></span></div><canvas id="c-lat"></canvas>
<div class="lbl"><span>CPU %</span><span id="v-cpu"></span></div><canvas id="c-cpu"></canvas>
<div class="lbl"><span>GPU %</span><span id="v-gpu"></span></div><canvas id="c-gpu"></canvas></div>
<div class="card"><h2>Device</h2><table id="dev"></table></div>
<div class="card"><h2>Camera</h2><table id="cam"></table></div>
</div></main>
<script>
const $=id=>document.getElementById(id);let cur=0;const hist={fps:[],lat:[],cpu:[],gpu:[]};const N=120;
const na=v=>(v===null||v===undefined||v==='')?'<small>unavailable</small>':v;
const row=(k,v)=>`<tr><td>${k}</td><td>${na(v)}</td></tr>`;
const ago=s=>s==null?null:(s<1?'now':s<60?s.toFixed(0)+' s ago':(s/60).toFixed(1)+' min ago');
function push(k,v){hist[k].push(v==null?null:+v);if(hist[k].length>N)hist[k].shift()}
function draw(id,k,max){const c=$(id);const W=c.width=c.clientWidth*devicePixelRatio,H=c.height=c.clientHeight*devicePixelRatio;const g=c.getContext('2d');g.clearRect(0,0,W,H);
 const d=hist[k];const vals=d.filter(x=>x!=null);if(!vals.length)return;const mx=max||Math.max(1,...vals)*1.15;g.strokeStyle='#5aa9ff';g.lineWidth=1.5*devicePixelRatio;g.beginPath();let pen=false;
 d.forEach((v,i)=>{const x=i/(N-1)*W;if(v==null){pen=false;return}const y=H-(v/mx)*(H-4)-2;pen?g.lineTo(x,y):g.moveTo(x,y);pen=true});g.stroke()}
function badge(el,txt,cls){el.textContent=txt;el.className='badge '+(cls||'')}
async function tick(){let s;try{s=await (await fetch('/api/status',{cache:'no-store'})).json()}catch(e){badge($('b-inf'),'live view: no connection','bad');return}
 const m=s.model||{},inf=s.inference||{},dev=s.device||{},sys=s.system||{},chans=s.channels||[];
 const c=chans[cur]||chans[0]||null,src=c?c.source:null,lat=c?c.latency_ms:{};
 document.title=`${m.project||'CamThink'} live view`;$('title').textContent=`${m.project||'CamThink'} · live view`;
 if(chans.length>1){$('chsel').style.display='';const sel=$('ch');if(sel.options.length!==chans.length){sel.innerHTML=chans.map(x=>`<option value="${x.channel}">${x.name}</option>`).join('');sel.value=cur}}
 if(src)badge($('b-cam'),src.connected?`camera connected (${src.decode_fps} fps)`:'camera disconnected',src.connected?'ok':'bad');
 else badge($('b-cam'),inf.mode==='rtsp'?'camera: waiting':'no camera configured','warn');
 badge($('b-inf'),'inference '+inf.state+(inf.state==='running'&&inf.mode!=='rtsp'?' (samples)':''),inf.state==='running'?(inf.mode==='rtsp'?'ok':'warn'):'bad');
 badge($('b-dev'),(dev.label||dev.hostname||'device')+(dev.name?' · '+dev.name:''),'');
 badge($('b-model'),`${m.project||'?'} ${m.model_version||''} · ${s.deployment||'no deployment'}`,'');
 badge($('b-view'),`${(s.liveview||{}).viewers||0} viewer(s)`,'');
 $('mdl').innerHTML=row('project',m.project)+row('model version',m.model_version)+row('deployment',m.deployment_id?`${m.deployment_id} (${m.target_ref})`:s.deployment)+row('export',m.export_id)
  +row('device',(dev.label||'')+(dev.name?' '+dev.name:'')+' · '+(dev.hostname||''))+row('input resolution',m.input_size?m.input_size+'×'+m.input_size:null)
  +row('precision',m.precision?m.precision+(m.tensorrt?' · TensorRT '+m.tensorrt:''):null)+row('confidence threshold',m.confidence!=null?m.confidence+(m.confidence_default!=null&&m.confidence!==m.confidence_default?` (deployment default ${m.confidence_default})`:''):null)
  +row('classes',(m.classes||[]).length?(m.classes||[]).map((n,i)=>`<span class="cls"><i class="sw" style="background:${(m.class_colours||[])[i]||'#888'}"></i>${n}</span>`).join(''):null);
 $('legend').innerHTML='';
 const last=c&&c.last_result_ts?(s.ts-c.last_result_ts):null;
 $('perf').innerHTML=row('detection fps (actual / target)',c?`${c.fps} / ${c.target_fps}`:null)+row('video input fps (decoded)',src?src.decode_fps:null)
  +row('inference latency p50 / p95 ms',lat&&lat.inference_p50!=null?`${lat.inference_p50} / ${lat.inference_p95}`:null)+row('pre / post-process p50 ms',lat&&lat.preprocess_p50!=null?`${lat.preprocess_p50} / ${lat.postprocess_p50}`:null)
  +row('decode→result latency p50 / p95 ms',lat&&lat.e2e_p50!=null?`${lat.e2e_p50} / ${lat.e2e_p95}`:null)
  +row('frames inferred',c?c.frames:null)+row('frames skipped (decoded, not inferred)',src?src.frames_dropped:null)+row('missed inference slots',c?c.missed_slots:null)
  +row('inference errors',c?c.errors:null)+row('last successful inference',last!=null?ago(last):null)
  +row('CPU %',sys.cpu_pct)+row('GPU %',sys.gpu_pct)+row('GPU clock MHz',sys.gpu_mhz)+row('RAM used / total MB',sys.ram_used_mb!=null?`${sys.ram_used_mb} / ${sys.ram_total_mb}`:null)
  +row('temperature CPU / GPU °C',sys.temp_cpu_c!=null||sys.temp_gpu_c!=null?`${sys.temp_cpu_c??'–'} / ${sys.temp_gpu_c??'–'}`:null)+row('power total mW',sys.power_total_mw)
  +row('inference process RSS MB',s.monitor?s.monitor.process_rss_mb:null)+row('live-view process RSS MB',(s.liveview||{}).process_rss_mb)
  +row('live-view encode p50 ms',s.monitor&&s.monitor.live?s.monitor.live.encode_ms_p50:null)+row('metrics source',s.system_backend);
 const d=c?c.detections_by_class:null;$('dets').innerHTML=d&&Object.keys(d).length?Object.entries(d).map(([k,v])=>{const i=(m.classes||[]).indexOf(k);return `<span class="cls"><i class="sw" style="background:${(m.class_colours||[])[i]||'#888'}"></i>${k}: <b>${v}</b></span>`}).join(''):(c?'<small>no detections in the last frame</small>':'<small>–</small>');
 $('dev').innerHTML=row('hostname',dev.hostname)+row('target',dev.name)+row('power mode',dev.power_mode)+row('runtime',(s.liveview||{}).runtime_version)+row('inference service',inf.state+(inf.reason?' – '+inf.reason:''))
  +row('heartbeat age s',inf.heartbeat_age_s)+row('config restart required',inf.config_restart_required?'yes (service restart)':'no')+row('live view uptime s',(s.liveview||{}).uptime_s);
 const cf=s.config||{};$('cam').innerHTML=(cf.sources||[]).map((x,i)=>row(x.name,x.url)).join('')+row('decode size',cf.inference?`${cf.inference.width}×${cf.inference.height}`:null)
  +row('inference fps target',cf.inference?cf.inference.fps:null)+row('display fps / quality',cf.liveview?`${cf.liveview.display_fps} / ${cf.liveview.quality}`:null)
  +row('decoder',src?src.backend:null)+row('reconnects',src?src.reconnects:null)+row('connected for s',src?src.connected_for_s:null)+row('disconnected for s',src?src.disconnected_for_s:null)+row('camera error',src?src.error:null);
 push('fps',c?c.fps:null);push('lat',lat?lat.inference_p50:null);push('cpu',sys.cpu_pct);push('gpu',sys.gpu_pct);
 $('v-fps').textContent=c?c.fps:'';$('v-lat').textContent=lat&&lat.inference_p50!=null?lat.inference_p50:'';$('v-cpu').textContent=sys.cpu_pct??'';$('v-gpu').textContent=sys.gpu_pct??'';
 draw('c-fps','fps',null);draw('c-lat','lat',null);draw('c-cpu','cpu',100);draw('c-gpu','gpu',100)}
$('ch').addEventListener('change',e=>{cur=+e.target.value;$('video').src='/stream.mjpg?ch='+cur+'&t='+Date.now()});
$('video').addEventListener('error',()=>setTimeout(()=>{$('video').src='/stream.mjpg?ch='+cur+'&t='+Date.now()},2000));
setInterval(tick,1000);tick();
</script></body></html>"""


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"camthink_edge/{VERSION}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet; errors still reach stderr via log_error
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                ch = max(int(q.get("ch", ["0"])[0] or 0), 0)
            except ValueError:
                ch = 0
            if url.path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", HTML.encode())
            if url.path in ("/api/status", "/stats.json"):
                return self._send(
                    200, "application/json", json.dumps(state.status(), default=str).encode()
                )
            if url.path == "/api/detections":
                _, meta = state.frame(ch)
                return self._send(
                    200 if meta else 404,
                    "application/json",
                    json.dumps(meta or {"error": state.no_video_reason(ch)}).encode(),
                )
            if url.path == "/healthz":
                return self._send(
                    200,
                    "application/json",
                    json.dumps({"ok": True, "runtime_version": VERSION}).encode(),
                )
            if url.path == "/snapshot.jpg":
                data = self._fresh_frame(ch, wait_s=1.5)
                if data is None:
                    data = state.placeholder(state.no_video_reason(ch))
                return self._send(200, "image/jpeg", data)
            if url.path == "/stream.mjpg":
                try:
                    fps = float(q.get("fps", [str(state.config()["liveview"]["display_fps"])])[0])
                except ValueError:
                    fps = 10.0
                return self._mjpeg(ch, fps)
            return self._send(404, "text/plain", b"not found")

        def _fresh_frame(self, ch: int, *, wait_s: float) -> bytes | None:
            """Ask the monitor for a frame (viewer marker) and wait briefly for a new one."""
            _, meta0 = state.frame(ch)
            seq0 = meta0.get("jpg_seq") if meta0 else None
            state.write_viewers(force_count=max(state.viewers, 1))
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                data, meta = state.frame(ch)
                if data and meta and meta.get("jpg_seq") != seq0:
                    return data
                time.sleep(0.05)
            data, meta = state.frame(ch)
            if data and state.jpg_fresh(meta):
                return data
            return None

        def _mjpeg(self, ch: int, fps: float) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            state.viewer_enter()
            period = 1.0 / max(min(fps, 60.0), 0.5)
            last_seq = None
            last_sent = 0.0
            last_meta_ts = time.monotonic()
            try:
                while True:
                    t0 = time.monotonic()
                    data, meta = state.frame(ch)
                    seq = meta.get("jpg_seq") if meta else None
                    fresh = data is not None and state.jpg_fresh(meta)
                    if fresh and seq != last_seq:
                        self._part(data)
                        last_seq, last_sent, last_meta_ts = seq, t0, t0
                    elif not fresh and t0 - last_meta_ts > 2.0 and t0 - last_sent > 1.0:
                        # no frames: show why, once a second, instead of a frozen picture
                        self._part(state.placeholder(state.no_video_reason(ch)))
                        last_sent = t0
                    dt = time.monotonic() - t0
                    if dt < period:
                        time.sleep(period - dt)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                state.viewer_leave()

        def _part(self, data: bytes) -> None:
            self.wfile.write(
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(data)).encode()
                + b"\r\n\r\n"
                + data
                + b"\r\n"
            )
            self.wfile.flush()
            state.frames_sent += 1
            state.bytes_sent += len(data)

    return Handler


def run_liveview(
    root: Path,
    *,
    port: int | None = None,
    bind: str | None = None,
    stop: threading.Event | None = None,
) -> int:
    root = Path(root)
    cfg = appcfg.load(root)
    port = int(port or cfg["liveview"]["port"])
    bind = bind or str(cfg["liveview"].get("bind") or "0.0.0.0")
    state = LiveState(root, port=port)
    state.live.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((bind, port), make_handler(state))
    httpd.daemon_threads = True
    stop = stop or threading.Event()

    def _sig(*_: Any) -> None:
        stop.set()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _sig)
        signal.signal(signal.SIGINT, _sig)
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    print(
        json.dumps({"liveview": "started", "bind": bind, "port": port, "root": str(root)}),
        flush=True,
    )
    try:
        while not stop.is_set():
            if state.viewers:
                state.write_viewers()  # keep the marker fresh while streams are open
            stop.wait(1.0)
    finally:
        state.write_viewers(force_count=0)
        httpd.shutdown()
    return 0
