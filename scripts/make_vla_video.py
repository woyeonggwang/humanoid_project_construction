"""8단계 영상 만들기: scripts/vla_pick_eval.py --video 가 남긴 영상에 명령 자막과 머리 카메라 화면을 넣는다.

- 입력: <폴더>/<이름>-step-0.mp4 (바깥에서 본 영상), <이름>_info.json (명령 문장, 팔을 움직인 구간), <이름>_head.mp4 (그 구간의 머리 카메라)
- 팔을 움직인 구간의 앞 --pre 초부터 뒤 --post 초까지만 잘라 쓴다 (받침대까지 걷는 부분은 뺌)
- 이름을 두 개 주면 좌우로 나란히 붙인 비교 영상을 만든다 (예: 스크립트 IK / VLA)
ffmpeg 는 imageio_ffmpeg 에 들어 있는 것을 쓴다.

예) python scripts\\make_vla_video.py --dir data\\vla_pick\\videos --names vla1_side --out portfolio\\g1_vla_pick_side.mp4 --gif portfolio\\g1_vla_pick_side.gif
    python scripts\\make_vla_video.py --dir data\\vla_pick\\videos --names script1_side,vla1_side --out portfolio\\g1_vla_pick_compare.mp4 --gif ...
"""

import argparse
import json
import os
import subprocess
import tempfile

import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, help="영상·정보 파일이 있는 폴더")
parser.add_argument("--names", required=True, help="영상 이름 (쉼표로 두 개까지)")
parser.add_argument("--out", required=True)
parser.add_argument("--gif", default=None)
parser.add_argument("--pre", type=float, default=1.5)
parser.add_argument("--post", type=float, default=1.0)
parser.add_argument("--gif_width", type=int, default=720)
parser.add_argument("--gif_fps", type=int, default=12)
parser.add_argument("--gif_colors", type=int, default=96)
parser.add_argument("--font", default=os.path.join(os.environ.get("WINDIR", ""), "Fonts", "malgun.ttf"), help="한글 글꼴 파일")
args = parser.parse_args()

ff = imageio_ffmpeg.get_ffmpeg_exe()
names = [n.strip() for n in args.names.split(",")]
tmp = tempfile.mkdtemp(prefix="vla_video_")


def esc(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:")


def text_file(name: str, text: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return esc(path)


two = len(names) == 2
pw, ph = (640, 360) if two else (1280, 720)  # 한 칸의 크기
inset = 150 if two else 280
fs_label, fs_cap = (20, 20) if two else (30, 30)
inputs, chain, panels = [], [], []
dur = 0.0
for k, name in enumerate(names):
    info = json.load(open(os.path.join(args.dir, f"{name}_info.json"), encoding="utf-8"))
    fps = info["fps"]
    start = max(info["arm_start_step"] / fps - args.pre, 0.0)
    end = info["arm_end_step"] / fps + args.post
    lead = info["arm_start_step"] / fps - start  # 잘라 낸 영상에서 팔이 움직이기 시작하는 시각
    dur = max(dur, end - start)
    inputs += ["-i", os.path.join(args.dir, f"{name}-step-0.mp4"), "-i", os.path.join(args.dir, f"{name}_head.mp4")]
    chain.append(f"[{2 * k}:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS,scale={pw}:{ph}[m{k}]")
    chain.append(f"[{2 * k + 1}:v]scale={inset}:{inset},setpts=PTS+{lead:.3f}/TB[h{k}]")
    chain.append(f"[m{k}][h{k}]overlay=W-w-12:12:enable='gte(t,{lead:.3f})':eof_action=pass[o{k}]")
    label = text_file(f"label{k}.txt", info["label"].split(" (")[0] + ("" if info["success"] else "  (실패)"))
    cap = text_file(f"cap{k}.txt", f"명령: {info['task']}")
    chain.append(f"[o{k}]drawtext=fontfile='{esc(args.font)}':textfile='{label}':fontcolor=white:fontsize={fs_label}:x=14:y=12:"
                 f"box=1:boxcolor=black@0.55:boxborderw=8,"
                 f"drawtext=fontfile='{esc(args.font)}':textfile='{cap}':fontcolor=0xFFE066:fontsize={fs_cap}:x=14:y=h-th-18:"
                 f"box=1:boxcolor=black@0.55:boxborderw=8[p{k}]")
    panels.append(f"[p{k}]")
if two:
    chain.append(f"{panels[0]}{panels[1]}hstack=inputs=2[out]")
    last = "out"
else:
    last = "p0"
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
subprocess.run([ff, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chain), "-map", f"[{last}]", "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", args.out], check=True)
print(f"mp4 {os.path.getsize(args.out) / 1e6:.2f}MB, {dur:.1f}초 → {args.out}")
if args.gif:
    vf = (f"fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={args.gif_colors}[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", args.out, "-filter_complex", vf, args.gif], check=True)
    print(f"gif {os.path.getsize(args.gif) / 1e6:.2f}MB ({args.gif_width}px, {args.gif_fps}fps) → {args.gif}")
