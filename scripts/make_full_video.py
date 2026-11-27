"""전체 연결 영상에 자막(한국어 명령, VLM 계획, 지금 실행 중인 단계)과 머리 카메라 화면을 넣는다.

scripts/vla_full_eval.py --video 가 만든 것을 읽는다: <이름>-step-0.mp4 (바깥에서 본 영상), <이름>_head.mp4 (팔 조작용 머리 카메라),
<이름>_info.json (명령·계획·단계 시간표), <이름>_vlm.png (VLM 에 준 사진).
ffmpeg 는 imageio_ffmpeg 에 들어 있는 것을 쓴다. 한글 글꼴 파일은 --font 로 바꿀 수 있다.

예) python scripts\\make_full_video.py --dir data\\vla_full\\videos --name full_one_side --out portfolio\\g1_full_one.mp4 --gif portfolio\\g1_full_one.gif
"""

import argparse
import json
import os
import subprocess
import tempfile

import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="영상·정보 파일이 있는 폴더")
parser.add_argument("--name", required=True)
parser.add_argument("--out", required=True, help="결과 mp4")
parser.add_argument("--gif", default=None, help="같이 만들 GIF 경로")
parser.add_argument("--gif_width", type=int, default=720)
parser.add_argument("--gif_fps", type=int, default=8)
parser.add_argument("--gif_speed", type=float, default=4.0)
parser.add_argument("--gif_colors", type=int, default=64)
parser.add_argument("--font", default=os.path.join(os.environ.get("WINDIR", ""), "Fonts", "malgun.ttf"),
                    help="한글 글꼴 파일 (기본: Windows 의 맑은 고딕)")
args = parser.parse_args()

ff = imageio_ffmpeg.get_ffmpeg_exe()
info = json.load(open(os.path.join(args.dir, f"{args.name}_info.json"), encoding="utf-8"))
fps = info["fps"]
total = info["total_steps"] / fps


def esc(path: str) -> str:
    """ffmpeg 필터 안에서 쓸 경로: 역슬래시를 슬래시로, 드라이브 뒤 ':' 는 이스케이프."""
    return path.replace("\\", "/").replace(":", "\\:")


def plan_lines(raw: str) -> list[str]:
    """계획 JSON 을 짧게: 상자 하나(걷기 → 집기 → 걷기 → 내려놓기)를 한 줄로."""
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except json.JSONDecodeError:
        return ["계획: (읽지 못함)"]
    if obj.get("error"):
        return [f'계획: {{"plan": [], "error": "{obj["error"]}"}}']
    parts = [f'{st.get("skill")}({st.get("target") or st.get("instruction", "")})' for st in obj.get("plan", [])]
    rows = [" → ".join(parts[k:k + 4]) for k in range(0, len(parts), 4)]
    return [("계획: " if k == 0 else "      ") + row for k, row in enumerate(rows)]


tmp = tempfile.mkdtemp(prefix="full_caption_")
texts = []  # (글, 색, 크기, 시작, 끝, y)
plan = plan_lines(info["plan"])
plan_y = 720 - 58 - 34 * len(plan)
texts.append((f"명령: {info['command']}", "white", 30, 0.0, total, plan_y - 50))
for k, row in enumerate(plan):
    texts.append((row, "0xFFE066", 20, 0.0, total, plan_y + 34 * k))
ev = info["events"]
for k, e in enumerate(ev):
    start = e["step"] / fps
    end = ev[k + 1]["step"] / fps if k + 1 < len(ev) else total
    if end > start:
        bad = e["text"].startswith("실패")
        texts.append((f"지금: {e['text']}", "0xFF8080" if bad else "0x9BE39B", 24, start, end, 720 - 50))
chain = ["[1:v]scale=250:250[head]", "[0:v][head]overlay=W-w-16:16[v0]", "[2:v]scale=250:250[photo]",
         "[v0][photo]overlay=16:16:enable='between(t,0,4)'[v1]"]
last = "v1"
labels = [("VLM 에 준 사진", 16, 270, 0.0, 4.0), ("머리 카메라", 1280 - 250 - 16, 270, 0.0, total)]
for k, (text, x, y, start, end) in enumerate(labels):
    path = os.path.join(tmp, f"l{k}.txt")
    open(path, "w", encoding="utf-8", newline="\n").write(text)
    chain.append(f"[{last}]drawtext=fontfile='{esc(args.font)}':textfile='{esc(path)}':fontcolor=white:fontsize=18:x={x}:y={y}:"
                 f"box=1:boxcolor=black@0.55:boxborderw=5:enable='between(t,{start},{end})'[lb{k}]")
    last = f"lb{k}"
for k, (text, color, size, start, end, y) in enumerate(texts):
    path = os.path.join(tmp, f"t{k}.txt")
    open(path, "w", encoding="utf-8", newline="\n").write(text)
    chain.append(f"[{last}]drawtext=fontfile='{esc(args.font)}':textfile='{esc(path)}':fontcolor={color}:fontsize={size}:x=24:y={y}:"
                 f"box=1:boxcolor=black@0.55:boxborderw=7:enable='between(t,{start:.2f},{end:.2f})'[d{k}]")
    last = f"d{k}"
inputs = ["-i", os.path.join(args.dir, f"{args.name}-step-0.mp4"), "-i", os.path.join(args.dir, f"{args.name}_head.mp4"),
          "-loop", "1", "-t", "4", "-i", os.path.join(args.dir, f"{args.name}_vlm.png")]
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
subprocess.run([ff, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chain), "-map", f"[{last}]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", args.out], check=True)
print(f"mp4 {os.path.getsize(args.out) / 1e6:.2f}MB, {total:.1f}초 → {args.out}")
if args.gif:
    vf = (f"setpts=PTS/{args.gif_speed},fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={args.gif_colors}[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", args.out, "-filter_complex", vf, args.gif], check=True)
    print(f"gif {os.path.getsize(args.gif) / 1e6:.2f}MB ({args.gif_width}px, {args.gif_fps}fps, {args.gif_speed}배속) → {args.gif}")
