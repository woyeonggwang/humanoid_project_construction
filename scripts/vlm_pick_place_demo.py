"""VLM 연결 데모: 머리 카메라 사진 + 한국어 명령 → VLM → 계획 JSON → 걷기 정책 v2 와 상자 집어 옮기기 기술로 실행.

장면: 받침대 3개(왼쪽 stand_L, 가운데 stand_C, 오른쪽 stand_R)와 색 상자. 명령을 하나씩 처리한다:
  1. 서 있는 로봇의 머리 카메라 사진을 저장
  2. 사진과 명령을 VLM 작업자(vlm/infer_worker.py, 별도 파인튜닝 환경의 파이썬으로 띄운 하위 프로세스)에 보내 계획을 받음
  3. 계획의 기술을 차례로 실행: pick + place → 상자 집어 옮기기 상태 머신, go_to → 받침대 앞까지 걷기, wait·stop → 제자리
     거부(error)면 움직이지 않음
결과: 명령마다 VLM 출력·걸린 시간·실행 결과, 사진, (--video 면) 영상과 자막용 시간표(JSON)

VLM 을 띄울 파이썬은 --vlm_python 또는 환경 변수 VLM_PYTHON 으로 준다 (Isaac 환경의 파이썬과 다른 것).
예) python -u scripts\\vlm_pick_place_demo.py --headless --video --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
    VLM 없이 실행기만 시험: ... --plans "[{\\"plan\\":[{\\"skill\\":\\"go_to\\",\\"target\\":\\"stand_R\\"}]}]" --commands "오른쪽 받침대로 가"
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Image + Korean command -> VLM plan -> pick-and-place execution.")
parser.add_argument("--layout", type=str, default="stand_L=red,stand_C=blue", help="처음 상자 배치 '받침대=색,...' (없는 받침대는 비어 있음)")
parser.add_argument("--commands", type=str,
                    default="빨간 상자를 오른쪽 받침대로 옮겨|노란 상자를 가운데 받침대로 옮겨|파란 상자를 빈 받침대로 옮겨",
                    help="한국어 명령들 ('|' 로 구분)")
parser.add_argument("--plans", type=str, default=None, help="VLM 대신 쓸 계획 JSON 목록 (실행기 시험용)")
parser.add_argument("--vlm_python", type=str, default=os.environ.get("VLM_PYTHON"), help="VLM 을 띄울 파이썬 실행 파일")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3.5-0.8B")
parser.add_argument("--vlm_adapter", type=str, default=os.path.join("outputs", "qwen35_08b", "lora"), help="vlm 폴더 기준 LoRA 어댑터")
parser.add_argument("--task", type=str, default="Humanoid-G1-VLM-Scene-v0")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--max_time", type=float, default=150.0, help="전체 최대 시간 (초)")
parser.add_argument("--video", action="store_true")
parser.add_argument("--name", type=str, default="vlm_demo")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # 머리 카메라

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vlm"))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry import pick_place_machine as ppm  # noqa: E402
from humanoid_tasks.g1_carry.pick_place_cfg import BOX_Z_ON_TABLE  # noqa: E402
from humanoid_tasks.g1_carry.vlm_scene import apply_layouts, read_camera  # noqa: E402
from humanoid_tasks.g1_carry.vlm_scene_cfg import STAND_NAMES  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from schema import ERRORS, OBJECTS, SKILLS  # noqa: E402

MARK = "@@VLM@@"
commands = [c.strip() for c in args_cli.commands.split("|") if c.strip()]
fixed_plans = json.loads(args_cli.plans) if args_cli.plans else None
boxes0 = {s: None for s in STAND_NAMES}
for item in args_cli.layout.split(","):
    if item.strip():
        stand, color = item.strip().split("=")
        boxes0[stand] = color

# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.seed = args_cli.seed
if args_cli.video:
    # 고정 카메라: 로봇 뒤 오른쪽 위에서 받침대 세 개와 로봇이 함께 보이게
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (-1.7, -2.3, 2.0)
    env_cfg.viewer.lookat = (1.0, 0.0, 0.65)
    env_cfg.viewer.resolution = (1280, 720)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
out_dir = os.path.join(os.path.dirname(resume_path), "videos", "vlm_demo")
os.makedirs(out_dir, exist_ok=True)
step_dt = env_cfg.sim.dt * env_cfg.decimation
max_steps = int(args_cli.max_time / step_dt)

env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
if args_cli.video:
    env = gym.wrappers.RecordVideo(env, video_folder=out_dir, step_trigger=lambda step: step == 0,
                                   video_length=max_steps, name_prefix=args_cli.name, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
device = u.device
robot = u.scene["robot"]
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
print(f"[demo] 걷기 정책: {os.path.join(os.path.basename(os.path.dirname(resume_path)), os.path.basename(resume_path))}")

machine = ppm.PickPlaceMachine(u)
cmd_term = u.command_manager.get_term("base_velocity")
scene = apply_layouts(u, [{"boxes": boxes0}])
stand_xy = {name: scene["stand_pos"][k, :, :2].clone() for k, name in enumerate(STAND_NAMES)}  # (1, 2)
obstacles = torch.stack([stand_xy[n] for n in STAND_NAMES])  # (3, 1, 2)
yaw0 = torch.zeros(1, device=device)  # 받침대는 모두 로봇 시작 방향(+x)을 마주 봄
home_xy = u.scene.env_origins[:1, :2].clone()  # 처음 자리 (사진을 찍는 곳)

# ---------------------------------------------------------------------------
# VLM 작업자
# ---------------------------------------------------------------------------
vlm = None
vlm_info = {}
if fixed_plans is None:
    if not args_cli.vlm_python:
        raise SystemExit("--vlm_python 또는 환경 변수 VLM_PYTHON 으로 VLM 을 띄울 파이썬을 알려 주세요")
    cmd_line = [args_cli.vlm_python, "-X", "utf8", "infer_worker.py", "--model", args_cli.vlm_model]
    if args_cli.vlm_adapter:
        cmd_line += ["--adapter", args_cli.vlm_adapter]
    vlm = subprocess.Popen(cmd_line, cwd=str(ROOT / "vlm"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=open(os.path.join(out_dir, "vlm_worker.log"), "w", encoding="utf-8"),
                           text=True, encoding="utf-8")


def vlm_read():
    while True:
        line = vlm.stdout.readline()
        if not line:
            raise RuntimeError("VLM 작업자가 끝났습니다 (vlm_worker.log 참고)")
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


gpu_before = gpu_used_mb()
if vlm is not None:
    vlm_info = vlm_read()
    print(f"[demo] VLM 준비: 불러오기 {vlm_info['load_s']}초, VLM 프로세스 GPU {vlm_info['gpu_mb']}MB, "
          f"GPU 전체 사용량 {gpu_before} → {gpu_used_mb()}MB")

# ---------------------------------------------------------------------------
# 실행 도우미
# ---------------------------------------------------------------------------
obs = None
step = 0
fell = False


def tick(n: int = 1):
    """n 스텝 진행. 상태 머신이 걷기 명령과 팔을 정한다 (머신이 쉬는 중이면 제자리)."""
    global obs, step, fell
    for _ in range(n):
        if step >= max_steps:
            return
        machine.pre_step()
        obs, _, _, _ = env.step(policy(obs))
        fell = fell or bool(u.termination_manager.terminated.any())
        cmd_term.vel_command_b[:] = machine.post_step(u.termination_manager.terminated)
        obs = env.get_observations()
        step += 1


def run_machine() -> bool:
    while bool(machine.active.any()) and step < max_steps:
        tick()
    return bool((machine.state == ppm.S_DONE).all())


def box_on_stand(color: str, stand: str) -> float:
    box = u.scene[f"{color}_box"]
    return float(torch.norm(box.data.root_pos_w[0, :2] - stand_xy[stand][0]))


def true_scene() -> dict:
    """지금 실제로 어느 받침대에 어떤 색 상자가 있는지 (받침대 가운데에서 8cm 안, 받침대 위 높이)."""
    out = {s: None for s in STAND_NAMES}
    for obj in OBJECTS:
        color = obj.replace("_box", "")
        z = float(u.scene[obj].data.root_pos_w[0, 2])
        for s in STAND_NAMES:
            if box_on_stand(color, s) < 0.08 and abs(z - BOX_Z_ON_TABLE) < 0.03:
                out[s] = color
    return out


def execute(plan: list) -> tuple[bool, str]:
    """계획의 기술을 차례로 실행. (성공 여부, 설명)"""
    i = 0
    while i < len(plan):
        sk = plan[i]
        name = sk.get("skill")
        if name not in SKILLS:
            return False, f"모르는 기술 {name}"
        if name == "pick":
            color, src = str(sk.get("object", "")).replace("_box", ""), sk.get("from")
            if f"{color}_box" not in OBJECTS or src not in stand_xy:
                return False, f"pick 인자가 잘못됨: {sk}"
            if i + 1 >= len(plan) or plan[i + 1].get("skill") != "place" or plan[i + 1].get("to") not in stand_xy:
                return False, "pick 뒤에 place(to) 가 없음 (이 데모는 집어서 옮기기만 실행)"
            dst = plan[i + 1]["to"]
            machine.start(u.scene[f"{color}_box"], stand_xy[src], yaw0, stand_xy[dst], yaw0, obstacles=obstacles)
            if not run_machine():
                where = ppm.STATE_NAMES[int(machine.fail_state[0])] if int(machine.fail_state[0]) >= 0 else "시간 초과"
                return False, f"{color} 상자 옮기기 실패 ({where})"
            err = box_on_stand(color, dst)
            if err > 0.05:
                return False, f"{color} 상자가 {dst} 가운데에서 {err * 100:.1f}cm"
            i += 2
            continue
        if name == "place":
            return False, "들고 있는 상자가 없음 (place 만 있는 계획)"
        if name == "go_to":
            if sk.get("target") not in stand_xy:
                return False, f"go_to 대상이 잘못됨: {sk}"
            machine.start_walk(stand_xy[sk["target"]], yaw0, obstacles=obstacles)
            if not run_machine():
                return False, f"{sk['target']} 로 걷기 실패"
        elif name == "wait":
            tick(int(float(sk.get("sec", 1)) / step_dt))
        elif name == "stop":
            tick(25)
        i += 1
    return (not fell), "넘어짐" if fell else "실행 완료"


# ---------------------------------------------------------------------------
# 명령 처리
# ---------------------------------------------------------------------------
timeline = []  # 자막용: 명령별 시작·끝 시각(초), 문장, VLM 출력, 사진
results = []
with torch.inference_mode():
    cmd_term.vel_command_b[:] = 0.0
    obs = env.get_observations()
    apply_layouts(u, [{"boxes": boxes0}])
    tick(75)  # 처음 자리 잡기
    for n, text in enumerate(commands):
        if step >= max_steps:
            break
        # 사진은 늘 처음 자리(받침대 세 개가 한 화면에 들어오는 곳)에서 찍는다: 앞 명령을 실행하느라 옮겨 갔으면 돌아옴
        away = float(torch.norm(robot.data.root_pos_w[0, :2] - home_xy[0]))
        if away > 0.10 or abs(float(ppm.wrap(robot.data.heading_w - yaw0)[0])) > 0.10:
            machine.start_walk(home_xy + torch.tensor([[ppm.STANCE_DIST, 0.0]], device=device), yaw0, obstacles=obstacles)
            if not run_machine():
                print(f"[demo] 처음 자리로 돌아가지 못함 (명령 {n + 1} 전)")
        tick(25)
        off = robot.data.root_pos_w[0, :2] - home_xy[0]
        print(f"[demo] 사진 찍는 자리: 처음 자리에서 앞 {float(off[0]) * 100:+.1f}cm·왼쪽 {float(off[1]) * 100:+.1f}cm, "
              f"방향 {math.degrees(float(ppm.wrap(robot.data.heading_w - yaw0)[0])):+.1f}°")
        before = true_scene()
        image_path = os.path.join(out_dir, f"{args_cli.name}_cmd{n + 1}.png")
        iio.imwrite(image_path, read_camera(u)[0].cpu().numpy())
        t_cmd = step * step_dt
        if vlm is not None:
            t0 = time.time()
            vlm.stdin.write(json.dumps({"image": os.path.abspath(image_path), "text": text}, ensure_ascii=False) + "\n")
            vlm.stdin.flush()
            ans = vlm_read()
            wall = time.time() - t0
            raw, parsed, latency = ans["raw"], ans["json"], ans["latency_s"]
        else:
            parsed, raw, latency, wall = fixed_plans[n], json.dumps(fixed_plans[n], ensure_ascii=False), 0.0, 0.0
        print(f"\n[demo] 명령 {n + 1}: {text}")
        print(f"[demo]   장면: {before}")
        print(f"[demo]   VLM 출력: {raw.strip()}  (생성 {latency:.2f}초, 왕복 {wall:.2f}초, GPU 전체 {gpu_used_mb()}MB)")
        tick(50)  # 자막을 읽을 시간
        if not isinstance(parsed, dict) or not isinstance(parsed.get("plan"), list):
            ok, note = False, "JSON 을 읽을 수 없음"
        elif parsed.get("error"):
            ok = parsed["error"] in ERRORS and parsed["plan"] == []
            note = f"거부: {parsed['error']} (움직이지 않음)"
            tick(200)  # 영상에서 거부 결과를 읽을 시간
        else:
            ok, note = execute(parsed["plan"])
        after = true_scene()
        print(f"[demo]   결과: {'성공' if ok else '실패'} — {note} | 실행 뒤 장면: {after}")
        results.append({"command": text, "vlm_raw": raw, "vlm_json": parsed, "latency_s": latency, "ok": ok, "note": note,
                        "scene_before": before, "scene_after": after})
        timeline.append({"start": round(t_cmd, 2), "end": round(step * step_dt, 2), "command": text,
                         "output": json.dumps(parsed, ensure_ascii=False) if parsed is not None else raw.strip(),
                         "image": os.path.basename(image_path), "ok": ok, "note": note})
    tick(50)

if vlm is not None:
    vlm.stdin.write(json.dumps({"quit": True}) + "\n")
    vlm.stdin.flush()
    vlm.wait(timeout=30)

with open(os.path.join(out_dir, f"{args_cli.name}_timeline.json"), "w", encoding="utf-8") as f:
    json.dump({"step_dt": step_dt, "steps": step, "items": timeline}, f, ensure_ascii=False, indent=1)
n_ok = sum(r["ok"] for r in results)
print(f"\n========== VLM 연결 데모: 명령 {len(results)}개 중 {n_ok}개 성공, 넘어짐 {'있음' if fell else '없음'}, {step * step_dt:.1f}초 ==========")
for r in results:
    print(f"  [{'성공' if r['ok'] else '실패'}] {r['command']} → {json.dumps(r['vlm_json'], ensure_ascii=False)} ({r['note']}, 생성 {r['latency_s']:.2f}초)")
rel = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, os.path.basename(os.path.dirname(resume_path)), "videos", "vlm_demo")
print(f"[demo] 사진·시간표: {rel}" + (f", 영상: {os.path.join(rel, args_cli.name + '-step-0.mp4')}" if args_cli.video else ""))

env.close()
simulation_app.close()
