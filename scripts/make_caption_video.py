"""VLM 연결 데모 영상에 자막(명령 문장과 VLM 출력 JSON)과 그때의 머리 카메라 사진을 넣는다.

scripts/vlm_pick_place_demo.py --video 가 만든 영상(<이름>-step-0.mp4)과 시간표(<이름>_timeline.json), 사진(<이름>_cmdN.png)을 읽는다.
ffmpeg 는 imageio_ffmpeg 에 들어 있는 것을 쓴다. 한글 글꼴 파일은 --font 로 바꿀 수 있다.

예) python scripts\\make_caption_video.py --dir <영상 폴더> --name vlm_demo --out portfolio\\g1_vlm_pick_place.mp4 --gif portfolio\\g1_vlm_pick_place.gif
"""

import argparse
import json
import os
import subprocess
import tempfile

import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="영상·시간표·사진이 있는 폴더")
parser.add_argument("--name", default="vlm_demo")
parser.add_argument("--out", required=True, help="결과 mp4")
parser.add_argument("--gif", default=None, help="같이 만들 GIF 경로")
parser.add_argument("--gif_width", type=int, default=720)
parser.add_argument("--gif_fps", type=int, default=8)
parser.add_argument("--gif_speed", type=float, default=3.0)
parser.add_argument("--gif_colors", type=int, default=64)
parser.add_argument("--font", default=os.path.join(os.environ.get("WINDIR", ""), "Fonts", "malgun.ttf"),
                    help="한글 글꼴 파일 (기본: Windows 의 맑은 고딕)")
args = parser.parse_args()

ff = imageio_ffmpeg.get_ffmpeg_exe()
video = os.path.join(args.dir, f"{args.name}-step-0.mp4")
timeline = json.load(open(os.path.join(args.dir, f"{args.name}_timeline.json"), encoding="utf-8"))
items = timeline["items"]


def esc(path: str) -> str:
    """ffmpeg 필터 안에서 쓸 경로: 역슬래시를 슬래시로, 드라이브 뒤 ':' 는 이스케이프."""
    return path.replace("\\", "/").replace(":", "\\:")


tmp =tempfile.mkdtemp(prefix="caption_")
inputs = ["-i", video]
chain, last = [], "0:v"
# 사진: 오른쪽 위에 그 명령을 처리하는 동안 표시
for k, it in enumerate(items):
    inputs += ["-i", os.path.join(args.dir, it["image"])]
    chain.append(f"[{k + 1}:v]scale=300:300[p{k}]")
    chain.append(f"[{last}][p{k}]overlay=W-w-20:20:enable='between(t,{it['start']},{it['end']})'[o{k}]")
    last = f"o{k}"
# 자막: 아래쪽에 명령(흰색)과 VLM 출력(노란색), 결과
for k, it in enumerate(items):
    lines = [(f"명령 {k + 1}: {it['command']}", "white", 30),
             (it["output"], "0xFFE066", 20),  # 한 줄 (기술 3개짜리 계획까지 화면 폭에 들어감)
             (("성공 — " if it["ok"] else "실패 — ") + it["note"], "0x9BE39B" if it["ok"] else "0xFF8080", 22)]
    y = 720 - 150
    for j, (text, color, size) in enumerate(lines):
        path = os.path.join(tmp, f"t{k}_{j}.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        start = it["start"] if j < 2 else max(it["start"], it["end"] - 3.0)  # 결과는 마지막 3초에만
        chain.append(f"[{last}]drawtext=fontfile='{esc(args.font)}':textfile='{esc(path)}':fontcolor={color}:fontsize={size}:"
                     f"x=30:y={y}:box=1:boxcolor=black@0.55:boxborderw=8:line_spacing=6:"
                     f"enable='between(t,{start},{it['end']})'[d{k}_{j}]")
        last = f"d{k}_{j}"
        y += size + 20
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
subprocess.run([ff, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chain), "-map", f"[{last}]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", args.out], check=True)
print(f"mp4 {os.path.getsize(args.out) / 1e6:.2f}MB → {args.out}")
if args.gif:
    vf = (f"setpts=PTS/{args.gif_speed},fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={args.gif_colors}[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", args.out, "-filter_complex", vf, args.gif], check=True)
    print(f"gif {os.path.getsize(args.gif) / 1e6:.2f}MB ({args.gif_width}px, {args.gif_fps}fps, {args.gif_speed}배속) → {args.gif}")
