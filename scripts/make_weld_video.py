"""용접 근접 영상에 자막(용접선, 지금 단계, 잰 값)과 머리 카메라 화면을 넣고, 용접 앞뒤만 잘라 낸다.

scripts/weld_eval.py --video 가 만든 것을 읽는다: <이름>-step-0.mp4 (바깥에서 본 영상), <이름>_head.mp4 (앞을 보는 머리 카메라),
<이름>_info.json (단계 시간표와 잰 값). ffmpeg 는 imageio_ffmpeg 에 들어 있는 것을 쓴다.

예) python scripts\\make_weld_video.py --dir data\\weld\\videos --name weld_b4 --out portfolio\\g1_weld_close.mp4 --gif portfolio\\g1_weld_close.gif
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "humanoid_tasks" / "g1_carry"))
import site_layout as sl  # noqa: E402  (isaaclab 을 쓰지 않는 모듈이라 바로 불러온다)

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--gif", default=None)
parser.add_argument("--gif_width", type=int, default=720)
parser.add_argument("--gif_fps", type=int, default=10)
parser.add_argument("--gif_speed", type=float, default=2.0)
parser.add_argument("--gif_colors", type=int, default=96)
parser.add_argument("--from_phase", default="팔 뻗기", help="이 단계가 시작되기 1초 전부터")
parser.add_argument("--to_phase", default="용접 끝", help="이 단계가 시작되고 1.5초 뒤까지")
parser.add_argument("--head_size", type=int, default=260)
parser.add_argument("--font", default=os.path.join(os.environ.get("WINDIR", ""), "Fonts", "malgun.ttf"))
args = parser.parse_args()

ff = imageio_ffmpeg.get_ffmpeg_exe()
info = json.load(open(os.path.join(args.dir, f"{args.name}_info.json"), encoding="utf-8"))
fps, total = info["fps"], info["total_steps"] / info["fps"]
ev = info["events"]
t0 = max(0.0, next((e["step"] for e in ev if e["text"] == args.from_phase), 0) / fps - 1.0)
t1 = min(total, next((e["step"] for e in ev if e["text"] == args.to_phase), info["total_steps"]) / fps + 1.5)


def esc(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:")


tmp = tempfile.mkdtemp(prefix="weld_caption_")
seam = info["seam"]
res = info["result"]
texts = [(f"용접: {sl.SEAM_KO.get(seam, seam)} ({seam})", "white", 28, t0, t1, 720 - 96)]
for k, e in enumerate(ev):
    start = e["step"] / fps
    end = ev[k + 1]["step"] / fps if k + 1 < len(ev) else total
    if end > t0 and start < t1:
        texts.append((f"지금: {e['text']}", "0xFF8080" if e["text"] == "실패" else "0x9BE39B", 24, max(start, t0), min(end, t1), 720 - 50))
if res.get("err_max_mm") is not None and res.get("welded"):
    t_w = next((e["step"] for e in ev if e["text"] == "가운데로"), None)
    if t_w is not None:
        texts.append((f"경로 오차 평균 {res['err_mean_mm']:.1f}mm · 최대 {res['err_max_mm']:.1f}mm, 속도 {res['speed_mean_mm_s'] / 10:.1f}cm/s, "
                      f"용접선을 덮은 비율 {res['coverage']:.0%}", "0xFFE066", 22, t_w / fps, t1, 720 - 140))
h = args.head_size
chain = [f"[1:v]scale={h}:{h}[head]", f"[0:v][head]overlay=W-w-16:16[v0]"]
last = "v0"
items = [("머리 카메라", 1280 - h - 16, h + 20, t0, t1, "white", 18)] + [(t, 24, y, a, b, c, sz) for t, c, sz, a, b, y in texts]
for k, (text, x, y, start, end, color, size) in enumerate(items):
    path = os.path.join(tmp, f"t{k}.txt")
    open(path, "w", encoding="utf-8", newline="\n").write(text)
    chain.append(f"[{last}]drawtext=fontfile='{esc(args.font)}':textfile='{esc(path)}':fontcolor={color}:fontsize={size}:x={x}:y={y}:expansion=none:"
                 f"box=1:boxcolor=black@0.55:boxborderw=6:enable='between(t,{start:.2f},{end:.2f})'[d{k}]")
    last = f"d{k}"
chain.append(f"[{last}]trim=start={t0:.2f}:end={t1:.2f},setpts=PTS-STARTPTS[out]")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
subprocess.run([ff, "-y", "-loglevel", "error", "-i", os.path.join(args.dir, f"{args.name}-step-0.mp4"),
                "-i", os.path.join(args.dir, f"{args.name}_head.mp4"), "-filter_complex", ";".join(chain), "-map", "[out]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", args.out], check=True)
print(f"mp4 {os.path.getsize(args.out) / 1e6:.2f}MB, {t1 - t0:.1f}초 → {args.out}")
if args.gif:
    vf = (f"setpts=PTS/{args.gif_speed},fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={args.gif_colors}[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", args.out, "-filter_complex", vf, args.gif], check=True)
    print(f"gif {os.path.getsize(args.gif) / 1e6:.2f}MB ({args.gif_width}px, {args.gif_fps}fps, {args.gif_speed}배속) → {args.gif}")
