"""Compare DA3 dense depth against Aria sparse semidense depth samples.

For each shared frame, metrics are evaluated only where the sparse map has a
valid sample. Exports per-frame and aggregate CSV/JSON metrics, plus a sampled
prediction scatter, error histogram, and per-frame comparison panels.

Example:
    python aria/compare_depth_maps.py outputs/aria_sparse/depth da3_depth_npz \
        --rgb-dir outputs/aria_sparse/images --output-dir outputs/depth_eval
"""

from __future__ import annotations

import argparse
import csv
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sparse_dir", type=Path, help="aria_depth_maps.py depth/ directory")
    p.add_argument("dense_dir", type=Path, help="da3_colmap.py per-image NPZ directory")
    p.add_argument("--rgb-dir", type=Path, help="Optional resized RGB frame directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-depth", type=float, default=20.0, help="Ignore larger/nonpositive depths")
    p.add_argument("--confidence-percentile", type=float, default=0.0,
                   help="Optionally keep DA3 samples at/above this per-frame percentile (0 disables)")
    p.add_argument("--make-plots", action="store_true", help="Save per-frame qualitative panels")
    p.add_argument("--max-plots", type=int, default=30)
    p.add_argument("--scale-aligned", action="store_true",
                   help="Also report errors after one median scale fit per frame; raw metric scores remain included")
    p.add_argument("--max-scatter-points", type=int, default=100000)
    p.add_argument("--visualize", action="store_true",
                   help="Open a linked, interactive browser viewer for all compared frames")
    return p.parse_args()


def find_npz_files(root):
    return {path.stem: path for path in root.rglob("*.npz")}


def load_depth(path):
    with np.load(path, allow_pickle=False) as z:
        key = next((k for k in ("depth", "depths") if k in z.files), None)
        if key is None:
            raise ValueError(f"{path}: missing depth array")
        depth = np.asarray(z[key], dtype=np.float32).squeeze()
        conf = np.asarray(z["conf"], dtype=np.float32).squeeze() if "conf" in z.files else None
    if depth.ndim != 2:
        raise ValueError(f"{path}: expected one 2D depth map, got {depth.shape}")
    return depth, conf


def metric_row(gt, pred):
    error = pred - gt
    abs_error = np.abs(error)
    ratio = np.maximum(gt / pred, pred / gt)
    return {
        "samples": int(gt.size),
        "mae_m": float(abs_error.mean()),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "abs_rel": float(np.mean(abs_error / gt)),
        "median_abs_error_m": float(np.median(abs_error)),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def save_panel(stem, gt_map, pred_map, rgb_path, valid, output_path, vmin, vmax):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    if rgb_path and rgb_path.is_file():
        import cv2

        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb)
    axes[0].imshow(np.ma.masked_where(~np.isfinite(gt_map), gt_map), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Sparse GT (metres)")
    axes[1].imshow(pred_map, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("DA3 depth (metres)")
    error_map = np.full(gt_map.shape, np.nan, dtype=np.float32)
    error_map[valid] = np.abs(pred_map[valid] - gt_map[valid])
    axes[2].imshow(error_map, cmap="magma")
    axes[2].set_title("Absolute error at GT samples")
    axes[3].scatter(gt_map[valid], pred_map[valid], s=3, alpha=0.35)
    lim = max(float(gt_map[valid].max()), float(pred_map[valid].max()))
    axes[3].plot([0, lim], [0, lim], "k--", linewidth=1)
    axes[3].set(xlabel="Sparse GT (m)", ylabel="DA3 (m)", title="Sampled depth agreement", xlim=(0, lim), ylim=(0, lim))
    for ax in axes[:3]:
        ax.axis("off")
    fig.suptitle(stem)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


INTERACTIVE_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aria sparse vs DA3 depth</title>
<style>
body{font:14px system-ui,sans-serif;margin:18px;background:#f4f6f8;color:#17202a}h1{font-size:22px;margin:0 0 12px}
.toolbar{display:flex;gap:10px;align-items:center;margin-bottom:12px}.toolbar button,.toolbar select{font:inherit;padding:6px 10px}
#readout{padding:9px 12px;background:#fff;border:1px solid #ccd3da;border-radius:5px;margin:8px 0 14px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:14px}.panel{background:white;border:1px solid #d5dbe0;border-radius:6px;padding:10px;min-width:0}
.panel h2{font-size:16px;margin:0 0 8px}.panel canvas{display:block;width:100%;height:auto;background:#101820;touch-action:none}.hint{color:#5d6d7e;font-size:12px;margin:5px 0 0}
.rankings{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:14px;margin-top:14px}.rankings table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;border-bottom:1px solid #e4e8eb;padding:5px}tbody tr{cursor:pointer}tbody tr:hover{background:#eaf2f8}.frame-meta{font-size:12px;color:#5d6d7e}
@media(max-width:750px){.grid,.rankings{grid-template-columns:1fr}}
</style></head><body>
<h1>Aria sparse depth vs DA3 dense depth</h1>
<div class="toolbar"><button id="prev">← Previous</button><select id="frame"></select><button id="next">Next →</button><span id="frame-meta" class="frame-meta"></span></div>
<div id="readout">Hover a pixel in the sparse map, dense map, or sampled agreement plot to inspect the corresponding values.</div>
<div class="grid">
 <section class="panel"><h2>Sparse Aria depth (m)</h2><canvas id="sparse"></canvas><p class="hint">Hover a pixel to link it to the dense map and sampled agreement.</p></section>
 <section class="panel"><h2>Sampled depth agreement (x: sparse, y: dense)</h2><canvas id="agreement"></canvas><p class="hint">Each point retains its image pixel location. Hover a sample to highlight that pixel in both maps.</p></section>
 <section class="panel"><h2>DA3 dense depth (m)</h2><canvas id="dense"></canvas></section>
 <section class="panel"><h2 id="correlation-title">Depth difference vs DA3 confidence</h2><canvas id="correlation"></canvas><p class="hint">Vertical axis is DA3 − sparse depth in metres.</p></section>
</div>
<div class="rankings">
 <section class="panel"><h2>Largest frame errors</h2><table><thead><tr><th>Frame</th><th>MAE (m)</th><th>AbsRel</th><th>Samples</th></tr></thead><tbody id="error-rank"></tbody></table></section>
 <section class="panel"><h2>Fewest sparse depth points</h2><table><thead><tr><th>Frame</th><th>Sparse points</th><th>Compared</th><th>MAE (m)</th></tr></thead><tbody id="sparse-rank"></tbody></table></section>
</div>
<script>
let W=504,H=504,MAX_DEPTH=20,frames=[],active=0,data=null,hoverPixel=null,samples=[],agreementPlot=null;
const $=id=>document.getElementById(id),ctx=id=>$(id).getContext('2d');
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function fmt(v){return Number.isFinite(v)?v.toFixed(3):'n/a'}
function color(v,max){if(!Number.isFinite(v))return [16,24,32,255];let t=Math.max(0,Math.min(1,v/max)),h=(1-t)*240;let i=Math.floor(h/60),f=h/60-i,p=0,q=1-f;let rgb=[[1,f,0],[q,1,0],[0,1,f],[0,q,1],[f,0,1],[1,0,q]][Math.min(i,5)];return [Math.round(rgb[0]*255),Math.round(rgb[1]*255),Math.round(rgb[2]*255),255]}
function drawMap(id,arr){let c=$(id);if(c.width!==W||c.height!==H){c.width=W;c.height=H}let g=ctx(id),im=g.createImageData(W,H);for(let i=0;i<arr.length;i++){let a=color(arr[i],MAX_DEPTH),j=i*4;im.data[j]=a[0];im.data[j+1]=a[1];im.data[j+2]=a[2];im.data[j+3]=Number.isFinite(arr[i])?255:0}g.putImageData(im,0,0);if(hoverPixel){let [x,y]=hoverPixel;g.strokeStyle='#ff00cc';g.lineWidth=1.5;g.beginPath();g.arc(x+.5,y+.5,5,0,2*Math.PI);g.stroke();g.strokeStyle='rgba(255,0,204,.7)';g.beginPath();g.moveTo(x+.5,0);g.lineTo(x+.5,H);g.moveTo(0,y+.5);g.lineTo(W,y+.5);g.stroke()}}
function chartBase(id,xlabel,ylabel,xmin,xmax,ymin,ymax){let c=$(id),g=ctx(id),w=Math.round(c.clientWidth*devicePixelRatio),h=Math.round(Math.max(280,c.clientWidth*.65)*devicePixelRatio);if(c.width!==w)c.width=w;if(c.height!==h)c.height=h;g.fillStyle='#fff';g.fillRect(0,0,w,h);let m={l:58,r:15,t:12,b:45},pw=w-m.l-m.r,ph=h-m.t-m.b,x=v=>m.l+(v-xmin)/(xmax-xmin)*pw,y=v=>m.t+ph-(v-ymin)/(ymax-ymin)*ph;g.strokeStyle='#52616b';g.lineWidth=1;g.beginPath();g.moveTo(m.l,m.t);g.lineTo(m.l,m.t+ph);g.lineTo(m.l+pw,m.t+ph);g.stroke();g.fillStyle='#253746';g.font=`${11*devicePixelRatio}px system-ui`;g.textAlign='center';for(let k=0;k<=4;k++){let v=xmin+(xmax-xmin)*k/4,px=x(v);g.beginPath();g.moveTo(px,m.t+ph);g.lineTo(px,m.t+ph+4*devicePixelRatio);g.stroke();g.fillText(v.toPrecision(3),px,m.t+ph+17*devicePixelRatio);let vy=ymin+(ymax-ymin)*k/4,py=y(vy);g.textAlign='right';g.fillText(vy.toPrecision(3),m.l-6*devicePixelRatio,py+4*devicePixelRatio);g.textAlign='center'}g.fillText(xlabel,m.l+pw/2,h-5*devicePixelRatio);g.save();g.translate(13*devicePixelRatio,m.t+ph/2);g.rotate(-Math.PI/2);g.fillText(ylabel,0,0);g.restore();return {g,m,pw,ph,w,h,x,y,xmin,xmax,ymin,ymax}}
function validSamples(){samples=[];for(let i=0;i<data.sparse.length;i++){let s=data.sparse[i],d=data.dense[i],c=data.conf[i];if(Number.isFinite(s)&&s>0&&Number.isFinite(d)&&d>0)samples.push({i,x:i%W,y:Math.floor(i/W),s,d,c})}}
function drawAgreement(){let max=MAX_DEPTH,p=agreementPlot=chartBase('agreement','Sparse depth (m)','DA3 depth (m)',0,max,0,max),g=p.g;g.strokeStyle='#888';g.setLineDash([5,4]);g.beginPath();g.moveTo(p.x(0),p.y(0));g.lineTo(p.x(max),p.y(max));g.stroke();g.setLineDash([]);g.fillStyle='rgba(30,105,190,.45)';for(const q of samples){if(q.s<=max&&q.d<=max){g.beginPath();g.arc(p.x(q.s),p.y(q.d),2.4*devicePixelRatio,0,2*Math.PI);g.fill()}}if(hoverPixel){let q=samples.find(q=>q.i===hoverPixel[1]*W+hoverPixel[0]);if(q){g.fillStyle='#f0009c';g.beginPath();g.arc(p.x(q.s),p.y(q.d),6*devicePixelRatio,0,2*Math.PI);g.fill()}}}
function drawCorrelation(){let diffs=samples.filter(q=>Number.isFinite(q.c));if(!diffs.length){$('correlation-title').textContent='Depth difference vs DA3 confidence';let p=chartBase('correlation','DA3 confidence','DA3 − sparse (m)',0,1,-5,5);p.g.fillStyle='#555';p.g.fillText('No confidence values in these NPZ files',p.m.l+20,p.m.t+25);return}let xmin=Infinity,xmax=-Infinity,ymin=-1,ymax=1,sumx=0,sumy=0;for(const q of diffs){let e=q.d-q.s;xmin=Math.min(xmin,q.c);xmax=Math.max(xmax,q.c);ymin=Math.min(ymin,e);ymax=Math.max(ymax,e);sumx+=q.c;sumy+=e}let mx=sumx/diffs.length,my=sumy/diffs.length,cov=0,vx=0,vy=0;for(const q of diffs){let dx=q.c-mx,dy=q.d-q.s-my;cov+=dx*dy;vx+=dx*dx;vy+=dy*dy}let corr=vx&&vy?cov/Math.sqrt(vx*vy):NaN;$('correlation-title').textContent=`Depth difference vs DA3 confidence (r = ${fmt(corr)})`;if(xmin===xmax)xmax=xmin+1;if(ymin===ymax)ymax=ymin+1;let p=chartBase('correlation','DA3 confidence','DA3 − sparse (m)',xmin,xmax,ymin,ymax),g=p.g;g.fillStyle='rgba(180,70,35,.45)';for(const q of diffs){g.beginPath();g.arc(p.x(q.c),p.y(q.d-q.s),2.4*devicePixelRatio,0,2*Math.PI);g.fill()}}
function render(){if(!data)return;drawMap('sparse',data.sparse);drawMap('dense',data.dense);drawAgreement();drawCorrelation();let f=frames[active];$('frame-meta').textContent=`${active+1}/${frames.length} · ${f.frame} · sparse points ${f.gt_points} · compared ${f.samples} · MAE ${fmt(f.mae_m)} m`}
function infoAt(x,y){if(!data||x<0||x>=W||y<0||y>=H)return;let i=y*W+x,s=data.sparse[i],d=data.dense[i],c=data.conf[i],diff=Number.isFinite(s)&&Number.isFinite(d)?d-s:NaN;hoverPixel=[x,y];$('readout').textContent=`pixel (${x}, ${y})  |  sparse depth: ${fmt(s)} m  |  dense depth: ${fmt(d)} m  |  dense − sparse: ${fmt(diff)} m  |  DA3 confidence: ${fmt(c)}`;render()}
function hookMap(id){$(id).addEventListener('mousemove',ev=>{if(id==='sparse'||id==='dense'){let r=$(id).getBoundingClientRect();infoAt(Math.floor((ev.clientX-r.left)/r.width*W),Math.floor((ev.clientY-r.top)/r.height*H))}})}
function hookAgreement(){let c=$('agreement');c.addEventListener('mousemove',ev=>{let r=c.getBoundingClientRect(),p=agreementPlot;if(!p)return;let mx=(ev.clientX-r.left)*devicePixelRatio,my=(ev.clientY-r.top)*devicePixelRatio,best=null,bd=100;for(const q of samples){if(q.s>MAX_DEPTH||q.d>MAX_DEPTH)continue;let dx=p.x(q.s)-mx,dy=p.y(q.d)-my,ds=dx*dx+dy*dy;if(ds<bd){bd=ds;best=q}}if(best&&bd<100){infoAt(best.x,best.y)}})}
function selectFrame(i){active=(i+frames.length)%frames.length;hoverPixel=null;let f=frames[active];$('frame').value=String(active);fetch('frames/'+encodeURIComponent(f.file)).then(r=>{if(!r.ok)throw Error('Could not load '+f.file);return r.arrayBuffer()}).then(buf=>{let a=new Float32Array(buf),n=W*H;if(a.length!==3*n)throw Error('Unexpected frame data size');data={sparse:a.slice(0,n),dense:a.slice(n,2*n),conf:a.slice(2*n)};validSamples();$('readout').textContent='Hover a pixel in a depth map or sampled agreement plot to inspect corresponding values.';render()}).catch(e=>{$('readout').textContent=e.message})}
function fillRanks(id,arr,cols){$(id).innerHTML=arr.map(f=>`<tr data-frame="${frames.indexOf(f)}"><td>${esc(f.frame)}</td>${cols(f)}</tr>`).join('');$(id).querySelectorAll('tr').forEach(tr=>tr.onclick=()=>selectFrame(Number(tr.dataset.frame)))}
fetch('index.json').then(r=>r.json()).then(j=>{W=j.width;H=j.height;MAX_DEPTH=j.max_depth;frames=j.frames;$('frame').innerHTML=frames.map((f,i)=>`<option value="${i}">${esc(f.frame)}</option>`).join('');fillRanks('error-rank',[...frames].sort((a,b)=>b.mae_m-a.mae_m).slice(0,30),f=>`<td>${fmt(f.mae_m)}</td><td>${fmt(f.abs_rel)}</td><td>${f.samples}</td>`);fillRanks('sparse-rank',[...frames].sort((a,b)=>a.gt_points-b.gt_points).slice(0,30),f=>`<td>${f.gt_points}</td><td>${f.samples}</td><td>${fmt(f.mae_m)}</td>`);$('frame').onchange=()=>selectFrame(Number($('frame').value));$('prev').onclick=()=>selectFrame(active-1);$('next').onclick=()=>selectFrame(active+1);for(const id of ['sparse','dense'])hookMap(id);hookAgreement();window.addEventListener('resize',render);selectFrame(0)}).catch(e=>{$('readout').textContent='Failed to load viewer data: '+e.message});
</script></body></html>"""


def write_interactive_viewer(output_dir, frame_records, max_depth):
    """Write a lightweight local HTML viewer and one binary payload per frame."""
    app_dir = output_dir / "interactive"
    frame_dir = app_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for record in frame_records:
        stem, sparse, dense, conf, row, gt_points = record
        filename = f"{len(frames):06d}.bin"
        if conf is None:
            conf = np.full(sparse.shape, np.nan, dtype=np.float32)
        payload = np.concatenate((sparse.ravel(), dense.ravel(), conf.ravel())).astype("<f4", copy=False)
        payload.tofile(frame_dir / filename)
        frames.append({
            "frame": stem,
            "file": filename,
            "mae_m": row["mae_m"],
            "abs_rel": row["abs_rel"],
            "samples": row["samples"],
            "gt_points": gt_points,
        })
    shape = frame_records[0][1].shape if frame_records else (504, 504)
    (app_dir / "index.json").write_text(json.dumps({"width": shape[1], "height": shape[0], "max_depth": max_depth, "frames": frames}, separators=(",", ":")))
    (app_dir / "index.html").write_text(INTERACTIVE_HTML)
    return app_dir


def launch_interactive_viewer(app_dir):
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(app_dir), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{server.server_port}/index.html"
    print(f"Interactive viewer: {url} (Ctrl-C to stop)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    args = parse_args()
    if args.max_depth <= 0 or not 0 <= args.confidence_percentile < 100:
        raise SystemExit("max-depth must be positive and confidence percentile in [0, 100)")
    sparse_files, dense_files = find_npz_files(args.sparse_dir), find_npz_files(args.dense_dir)
    shared = sorted(sparse_files.keys() & dense_files.keys())
    if not shared:
        raise SystemExit("No matching NPZ basenames found in sparse and dense directories")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel_dir = output / "frames"
    if args.make_plots:
        panel_dir.mkdir(parents=True, exist_ok=True)

    rows, all_gt, all_pred, aligned_gt, aligned_pred = [], [], [], [], []
    all_confidence, all_confidence_errors, frame_records = [], [], []
    for stem in shared:
        gt_map, _ = load_depth(sparse_files[stem])
        pred_map, conf = load_depth(dense_files[stem])
        if pred_map.shape != gt_map.shape:
            raise ValueError(f"{stem}: sparse shape {gt_map.shape} != dense shape {pred_map.shape}")
        gt_valid = np.isfinite(gt_map) & (gt_map > 0) & (gt_map <= args.max_depth)
        prediction_valid = np.isfinite(pred_map) & (pred_map > 0) & (pred_map <= args.max_depth)
        valid = gt_valid & prediction_valid
        if conf is not None and conf.shape != pred_map.shape:
            raise ValueError(f"{stem}: confidence shape {conf.shape} != depth shape {pred_map.shape}")
        if conf is not None:
            corr_valid = valid & np.isfinite(conf)
            all_confidence.append(conf[corr_valid].astype(np.float64))
            all_confidence_errors.append((pred_map[corr_valid] - gt_map[corr_valid]).astype(np.float64))
        if args.confidence_percentile and conf is not None:
            finite_conf = conf[np.isfinite(conf)]
            if finite_conf.size:
                valid &= np.isfinite(conf) & (conf >= np.percentile(finite_conf, args.confidence_percentile))
        if not valid.any():
            print(f"{stem}: no valid overlapping samples")
            continue
        gt, pred = gt_map[valid].astype(np.float64), pred_map[valid].astype(np.float64)
        stats = metric_row(gt, pred)
        gt_points = int(gt_valid.sum())
        row = {"frame": stem, "coverage": float(valid.sum() / max(1, gt_points)), "gt_points": gt_points, **stats}
        if args.scale_aligned:
            scale = float(np.median(gt / pred))
            row["median_scale"] = scale
            scaled = pred * scale
            aligned_gt.append(gt)
            aligned_pred.append(scaled)
            for key, value in metric_row(gt, scaled).items():
                if key != "samples":
                    row[f"scale_aligned_{key}"] = value
        rows.append(row)
        all_gt.append(gt)
        all_pred.append(pred)
        if args.visualize:
            frame_records.append((stem, gt_map.astype(np.float32), pred_map.astype(np.float32),
                                 conf.astype(np.float32) if conf is not None else None, row, gt_points))
        if args.make_plots and len(rows) <= args.max_plots:
            rgb_path = args.rgb_dir / f"{stem}.jpg" if args.rgb_dir else None
            save_panel(stem, gt_map, pred_map, rgb_path, valid, panel_dir / f"{stem}.png",
                       0, args.max_depth)
        print(f"{stem}: {stats['samples']} samples, MAE={stats['mae_m']:.3f}m, AbsRel={stats['abs_rel']:.3f}")

    if not rows:
        raise SystemExit("No frames had valid overlapping sparse and dense depth values")
    gt_all, pred_all = np.concatenate(all_gt), np.concatenate(all_pred)
    summary = {
        "frames_compared": len(rows),
        "samples": int(gt_all.size),
        "metrics_raw": metric_row(gt_all, pred_all),
        "mean_frame_metrics_raw": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("mae_m", "rmse_m", "abs_rel", "median_abs_error_m", "delta1", "coverage")
        },
        "interpretation": "Scores are computed only at valid semidense samples; they measure agreement with Aria semidense points, which are sparse and may contain reconstruction error.",
    }
    if args.scale_aligned:
        summary["metrics_per_frame_scale_aligned"] = metric_row(np.concatenate(aligned_gt), np.concatenate(aligned_pred))
    if all_confidence:
        confidence_values = np.concatenate(all_confidence)
        difference_values = np.concatenate(all_confidence_errors)
        if confidence_values.size > 1 and np.std(confidence_values) > 0 and np.std(difference_values) > 0:
            corr = float(np.corrcoef(confidence_values, difference_values)[0, 1])
        else:
            corr = None
        summary["confidence_vs_depth_difference"] = {
            "pearson_r": corr,
            "samples": int(confidence_values.size),
            "difference": "DA3 dense depth minus sparse Aria depth, in metres",
        }
    write_csv(output / "per_frame_metrics.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    if gt_all.size > args.max_scatter_points:
        idx = rng.choice(gt_all.size, args.max_scatter_points, replace=False)
        plot_gt, plot_pred = gt_all[idx], pred_all[idx]
    else:
        plot_gt, plot_pred = gt_all, pred_all
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    axes[0, 0].scatter(plot_gt, plot_pred, s=3, alpha=0.25)
    axes[0, 0].plot([0, args.max_depth], [0, args.max_depth], "k--")
    axes[0, 0].set(xlabel="Sparse Aria depth (m)", ylabel="DA3 depth (m)", title="Depth agreement", xlim=(0, args.max_depth), ylim=(0, args.max_depth))
    axes[1, 0].hist(pred_all - gt_all, bins=100)
    axes[1, 0].set(xlabel="DA3 − sparse depth (m)", ylabel="Samples", title="Signed error distribution")
    axes[1, 1].scatter([r["gt_points"] for r in rows], [r["mae_m"] for r in rows], s=12, alpha=0.6)
    axes[1, 1].set(xlabel="Sparse GT points per frame", ylabel="MAE (m)", title="Frame error vs sparse sample count")
    if all_confidence and any(values.size for values in all_confidence):
        confidence_all = np.concatenate(all_confidence)
        confidence_errors = np.concatenate(all_confidence_errors)
        if confidence_all.size > args.max_scatter_points:
            idx = rng.choice(confidence_all.size, args.max_scatter_points, replace=False)
            confidence_all, confidence_errors = confidence_all[idx], confidence_errors[idx]
        axes[0, 1].scatter(confidence_all, confidence_errors, s=3, alpha=0.25)
        corr = summary["confidence_vs_depth_difference"]["pearson_r"]
        corr_label = f"r={corr:.3f}" if corr is not None else "r undefined"
        axes[0, 1].set(xlabel="DA3 confidence", ylabel="DA3 − sparse depth (m)", title=f"Depth difference vs confidence ({corr_label})")
    else:
        axes[0, 1].text(0.5, 0.5, "No DA3 confidence values available", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_axis_off()
    largest_errors = sorted(rows, key=lambda row: row["mae_m"], reverse=True)[:20]
    least_points = sorted(rows, key=lambda row: row["gt_points"])[:20]
    axes[2, 0].barh([r["frame"] for r in reversed(largest_errors)], [r["mae_m"] for r in reversed(largest_errors)])
    axes[2, 0].set(xlabel="MAE (m)", title="Frames with largest errors")
    axes[2, 1].barh([r["frame"] for r in reversed(least_points)], [r["gt_points"] for r in reversed(least_points)])
    axes[2, 1].set(xlabel="Valid sparse depth points", title="Frames with fewest sparse points")
    fig.savefig(output / "aggregate_plots.png", dpi=150)
    plt.close(fig)
    write_csv(output / "least_sparse_frames.csv", least_points)
    if args.visualize:
        app_dir = write_interactive_viewer(output, frame_records, args.max_depth)
        launch_interactive_viewer(app_dir)
    print(f"Compared {len(rows)} frames / {gt_all.size:,} samples. Results: {output}")


if __name__ == "__main__":
    main()
