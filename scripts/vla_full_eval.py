"""전체 연결: 한국어 명령 → VLM 이 단계로 나눈 계획 → 걷기 정책 + 팔(VLA 또는 스크립트 IK)로 실행하고 평가한다.

장면: 집는 받침대(색 상자 두 개)와 내려놓을 받침대(색 판 두 개). humanoid_tasks/g1_carry/vla_full.py
한 시나리오: 처음 자리에서 VLM 전용 카메라로 사진 한 장 → VLM 이 계획 JSON → 실행기가 단계를 차례로 실행 (한 단계가 실패하면 그 자리에서 멈춤, 재시도 없음)
  go_to pick_stand   팔을 벌린 자세로 → 집는 받침대 앞까지 걷고 멈춤 (pick_place_machine 의 걷기)
  pick               팔: --mode vla 는 VLA(문장 = 계획의 instruction), --mode script 는 스크립트 IK. 그 뒤 스크립트가 뒤로 물러나 상자를 양손 가운데에 둔 든 자세로 옮김
                     (든 자세는 내려놓기 시범의 시작 자세와 같은 관절값 vla_place.CARRY_Q)
  go_to place_stand  상자를 든 채 내려놓을 받침대 앞까지 걷고 멈춤
  place              팔: VLA 또는 스크립트 IK. 그 뒤 뒤로 물러나 팔을 기본 자세로 내림
시나리오 종류: one(상자 하나 옮기기), two(상자 둘을 서로 다른 판으로), refuse(거부해야 하는 명령). 장면·명령은 시드로 정해진다.
VLM 의 답은 --plans_out 으로 저장해 두고 --plans_from 으로 다시 쓸 수 있다 (같은 계획으로 VLA 와 스크립트 IK 를 비교).

별도 환경의 파이썬: --vla_python / VLA_PYTHON (VLA 작업자), --vlm_python / VLM_PYTHON (VLM 작업자)
예) python -u scripts\\vla_full_eval.py --headless --mode vla --ckpt <체크포인트>/pretrained_model --plans_out data\\vla_full\\plans.json --out data\\vla_full\\eval_vla.json --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run Korean command -> VLM plan -> walking policy + VLA arms, and evaluate.")
parser.add_argument("--mode", choices=["vla", "script"], default="vla", help="팔을 움직이는 쪽")
parser.add_argument("--ckpt", type=str, default=None, help="VLA 체크포인트의 pretrained_model 폴더")
parser.add_argument("--vla_python", type=str, default=os.environ.get("VLA_PYTHON"))
parser.add_argument("--vlm_python", type=str, default=os.environ.get("VLM_PYTHON"))
parser.add_argument("--vlm_adapter", type=str, default=os.path.join("vlm", "outputs", "qwen35_08b_full", "lora"))
parser.add_argument("--plans_from", type=str, default=None,
                    help="이전 실행이 저장한 VLM 답을 다시 씀 (VLM 을 띄우지 않음). 'expected' 면 정답 계획을 그대로 씀 (실행기 시험용)")
parser.add_argument("--plans_out", type=str, default=None, help="VLM 답을 저장할 파일")
parser.add_argument("--n_action_steps", type=int, default=50)
parser.add_argument("--n_one", type=int, default=24, help="상자 하나 옮기기 시나리오 수")
parser.add_argument("--n_two", type=int, default=24, help="상자 둘 옮기기 시나리오 수")
parser.add_argument("--n_refuse", type=int, default=12, help="거부해야 하는 명령 시나리오 수")
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--max_arm_time", type=float, default=12.0, help="집기·내려놓기 한 번의 제한 시간 (초)")
parser.add_argument("--task", type=str, default="Humanoid-G1-VLA-Full-v0")
parser.add_argument("--load_run", type=str, default=".*_loco_v2")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--seed", type=int, default=7000)
parser.add_argument("--out", type=str, default=None, help="결과 JSON 을 저장할 파일")
parser.add_argument("--video", action="store_true", help="환경 0 을 녹화 (num_envs 1, 시나리오 하나로)")
parser.add_argument("--view", choices=["side", "top"], default="side")
parser.add_argument("--video_dir", type=str, default=os.path.join("data", "vla_full", "videos"))
parser.add_argument("--name", type=str, default="vla_full")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱이 켜진 뒤 import ----
import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vlm"))
import isaaclab_tasks  # noqa: E402, F401
import humanoid_tasks.g1_carry  # noqa: E402, F401
from humanoid_tasks.g1_carry import pick_place_machine as ppm  # noqa: E402
from humanoid_tasks.g1_carry import vla_full as vf  # noqa: E402
from humanoid_tasks.g1_carry import vla_pick as vp  # noqa: E402
from humanoid_tasks.g1_carry import vla_place as vpl  # noqa: E402
from humanoid_tasks.g1_carry.vla_pick_cfg import SMALL_BOX_SIZE  # noqa: E402
from humanoid_tasks.g1_carry.vla_worker import VlaWorker, VlmWorker  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from schema import instruction_color  # noqa: E402
from templates_full import Gen  # noqa: E402

N = args_cli.num_envs
SIDES = (1, -1)
HALF = SMALL_BOX_SIZE[1] / 2


def gpu_used_mb() -> int:
    """GPU 전체 사용량 (MB). 다른 프로그램이 쓰는 몫도 들어 있음."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# 시나리오 (시드로 정해짐. 렌더·모델과 무관)
# ---------------------------------------------------------------------------
def make_scenarios(seed: int) -> list[dict]:
    rng = random.Random(seed)
    gen = Gen(rng, "all")  # 문장 패턴은 학습 때 본 것과 처음 보는 것을 모두 씀
    out = []

    def add(kind, template, outcome, on_pad):
        for _ in range(200):
            scene = vf.sample_scene(rng, n_boxes=2, on_pad=on_pad)
            row = gen.make(template, vf.describe(scene), outcome)
            if row is not None:
                out.append({"kind": kind, "scene": scene, "command": row["input"], "expected": row["output"],
                            "template": template, "outcome": row.get("outcome"), "pairs": row.get("pairs")})
                return
        raise RuntimeError(f"시나리오를 만들지 못함: {template} {outcome}")

    for _ in range(args_cli.n_one):
        add("one", "move1", "ok", False)
    for _ in range(args_cli.n_two):
        add("two", "move2", "ok", False)
    refuse = [("move1", "not_found", False), ("move1", "occupied", True), ("move2", "not_found", False),
              ("move2", "occupied", True), ("unsupported", None, False)]
    for i in range(args_cli.n_refuse):
        add("refuse", *refuse[i % len(refuse)])
    return out


def normalize(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("plan", []), list):
        return None
    return {"plan": obj.get("plan", []), "error": obj.get("error")}


scenarios = make_scenarios(args_cli.seed)
for i, sc in enumerate(scenarios):
    sc["id"] = i
rounds = []
for kind in ("one", "two", "refuse"):
    group = [sc for sc in scenarios if sc["kind"] == kind]
    for k in range(0, len(group), N):
        rounds.append(group[k:k + N])
print(f"[full] 시나리오 {len(scenarios)}개 (하나 옮기기 {args_cli.n_one}, 둘 옮기기 {args_cli.n_two}, 거부 {args_cli.n_refuse}), 묶음 {len(rounds)}개")

# ---------------------------------------------------------------------------
# 환경·정책·작업자
# ---------------------------------------------------------------------------
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
env_cfg.seed = args_cli.seed
if args_cli.video:
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.6, -4.4, 2.0) if args_cli.view == "side" else (0.95, -0.02, 5.4)
    env_cfg.viewer.lookat = (1.05, 0.0, 0.65) if args_cli.view == "side" else (0.95, 0.0, 0.0)
    env_cfg.viewer.resolution = (1280, 720)
agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
step_dt = env_cfg.sim.dt * env_cfg.decimation
T_ARM = int(args_cli.max_arm_time / step_dt)
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
if args_cli.video:
    os.makedirs(args_cli.video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(env, video_folder=args_cli.video_dir, step_trigger=lambda step: step == 0,
                                   video_length=9000, name_prefix=args_cli.name, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped
device = u.device
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=device)
mem = {"isaac": gpu_used_mb()}

machine = ppm.PickPlaceMachine(u, until="stand")
lift = vp.LiftTracker(machine)
ptrack = vpl.PlaceTracker(machine)
vexec = vp.VlaArmExecutor(machine)
seq_pick = vp.SidePickScript(machine)
seq_place = vpl.SidePlaceScript(machine)
cmd_term = u.command_manager.get_term("base_velocity")
light_rng = random.Random(args_cli.seed + 1)
tmp_dir = tempfile.mkdtemp(prefix="vla_full_")

vla = None
if args_cli.mode == "vla":
    if not args_cli.vla_python or not args_cli.ckpt:
        raise SystemExit("--mode vla 에는 --ckpt 와 --vla_python(또는 환경 변수 VLA_PYTHON)이 필요합니다")
    vla = VlaWorker(args_cli.vla_python, args_cli.ckpt)
    print(f"[full] VLA 작업자: 불러오기 {vla.info['load_s']}초, GPU {vla.info['gpu_mb']}MB")
    mem["+vla"] = gpu_used_mb()
plans_cache, vlm = {}, None
if args_cli.plans_from == "expected":  # 실행기만 시험할 때: 정답 계획을 VLM 의 답으로 씀
    plans_cache = {sc["id"]: {"raw": sc["expected"], "json": json.loads(sc["expected"]), "latency_s": 0.0} for sc in scenarios}
elif args_cli.plans_from:
    plans_cache = {int(k): v for k, v in json.load(open(args_cli.plans_from, encoding="utf-8")).items()}
else:
    if not args_cli.vlm_python:
        raise SystemExit("VLM 을 띄우려면 --vlm_python(또는 환경 변수 VLM_PYTHON)이 필요합니다 (또는 --plans_from)")
    vlm = VlmWorker(args_cli.vlm_python, args_cli.vlm_adapter)
    print(f"[full] VLM 작업자: 불러오기 {vlm.info['load_s']}초, GPU {vlm.info['gpu_mb']}MB")
    mem["+vlm"] = gpu_used_mb()
mem["max"] = max(v for v in mem.values())


class OtherSlot:
    """집을 때 '집으면 안 되는 상자'(다른 자리의 상자)의 잡는 위치. 없으면 아주 먼 값."""

    def __init__(self):
        self.color = [None] * N
        self.off = torch.zeros(N, 2, 3, device=device)
        self.off[..., 0], self.off[..., 2] = ppm.GRASP_X, ppm.GRASP_Z

    def set(self, e: int, color, target_side: int):
        self.color[e] = color
        # 다른 상자는 반대쪽 자리에 있으므로 안쪽·바깥 손이 바뀐다
        self.off[e, 0, 1] = HALF + (vp.GAP_INNER if target_side > 0 else vp.GAP_OUTER)
        self.off[e, 1, 1] = -(HALF + (vp.GAP_OUTER if target_side > 0 else vp.GAP_INNER))

    def hand_error(self) -> torch.Tensor:
        pos, yaw = torch.zeros(N, 3, device=device), torch.zeros(N, device=device)
        has = torch.zeros(N, dtype=torch.bool, device=device)
        for e, c in enumerate(self.color):
            if c:
                box = u.scene[f"small_{c}_box"]
                pos[e], yaw[e], has[e] = box.data.root_pos_w[e], ppm.yaw_of(box.data.root_quat_w)[e], True
        err = torch.norm(machine.ik.hand_pos_w() - machine.grasp_points_of(pos, yaw, self.off), dim=-1).max(dim=1).values
        return torch.where(has, err, torch.full_like(err, 1.0e9))


# 실행기의 환경별 단계
NEXT, WALK, PICK, BACK, WAIT, RECENTER, PLACE, HOME, DONE, FAIL = range(10)
WAIT_STEPS, RECENTER_STEPS, GIVE_UP_STEPS = 60, 75, 150
CARRY_OFF = torch.tensor([-ppm.GRASP_X, 0.0, -ppm.GRASP_Z], device=device)  # 든 자세에서 두 손 가운데 기준 상자 자리
CARRY_Q = torch.tensor(vpl.CARRY_Q, device=device)  # 든 자세의 팔 관절 목표값 (내려놓기 시범의 시작 자세와 같게)

obs = None
n_steps = 0
head_frames, events = [], []


def note(text: str):
    """녹화용: 지금 단계 이름을 시간표에 적는다 (환경 0)."""
    if args_cli.video:
        events.append({"step": n_steps, "text": text})


def one(e: int) -> torch.Tensor:
    m = torch.zeros(N, dtype=torch.bool, device=device)
    m[e] = True
    return m


records, stage_stats = [], {k: [0, 0] for k in ("go_to", "pick", "place")}
vla_latency = []
t_start = time.time()
with torch.inference_mode():
    for r_i, group in enumerate(rounds):
        real = len(group)
        group = group + [group[-1]] * (N - real)  # 마지막 묶음의 남는 환경은 같은 시나리오로 채우고 세지 않음
        env.reset()
        machine.reset()
        seq_pick.reset()
        seq_place.reset()
        lay = vf.apply_scenes(u, [sc["scene"] for sc in group])
        vp.random_lighting(light_rng)
        cmd_term.vel_command_b[:] = 0.0
        obs = env.get_observations()
        for _ in range(25):  # 자리 잡기
            machine.pre_step()
            n_steps += 1
            obs, _, _, _ = env.step(policy(obs))
            cmd_term.vel_command_b[:] = machine.post_step(u.termination_manager.terminated.clone())
            obs = env.get_observations()
        # ---- VLM: 처음 자리의 사진 한 장 + 명령 → 계획 ----
        photo = vf.read_vlm_camera(u).cpu().numpy()
        plans, vlm_ok, preds = [], [], []
        for e, sc in enumerate(group):
            if sc["id"] not in plans_cache:
                path = os.path.join(tmp_dir, f"vlm_{sc['id']}.png")
                iio.imwrite(path, photo[e])
                ans = vlm.ask(path, sc["command"])
                plans_cache[sc["id"]] = {"raw": ans["raw"], "json": ans["json"], "latency_s": ans["latency_s"]}
            pred = normalize(plans_cache[sc["id"]]["json"])
            preds.append(pred)
            vlm_ok.append(pred is not None and pred == normalize(json.loads(sc["expected"])))
            plans.append(pred["plan"] if pred and not pred["error"] else [])
        if args_cli.video:
            iio.imwrite(os.path.join(args_cli.video_dir, f"{args_cli.name}_vlm.png"), photo[0])
        # ---- 실행 준비 ----
        slot = [dict(sc["scene"]["boxes"]) for sc in group]  # 집는 받침대의 자리별 상자 색 (집으면 None)
        pads = [dict(sc["scene"]["pads"]) for sc in group]
        pad_box = [dict(sc["scene"]["on_pad"]) for sc in group]  # 판 위의 상자
        held = [None] * N
        boxes = vp.BoxGroup(u, [next((c for c in s.values() if c), "red") for s in slot])
        machine.set_grasp_geometry(SMALL_BOX_SIZE, vp.GAP_OUTER, vp.GAP_OUTER)
        other = OtherSlot()
        place_scene = vpl.PlaceScene(boxes, lay.place_xy, lay.place_yaw, lay.pad_xy, torch.ones(N, device=device))
        obstacles = torch.stack([lay.pick_xy, lay.place_xy])
        phase = [NEXT if plans[e] else DONE for e in range(N)]
        idx, t_ph, after, loc, walk_to = [0] * N, [0] * N, [NEXT] * N, ["home"] * N, [None] * N
        fail_at, t_done = [None] * N, [None] * N
        rec_from = torch.zeros(N, 4, device=device)  # 든 자세로 옮길 때의 처음 상자 자리 (att_off 3 + att_yaw)
        rec_q = torch.zeros(N, 10, device=device)  # 그때의 처음 팔 관절 목표값
        chunk = torch.zeros(N, 50, vp.ACTION_DIM, device=device)
        k_chunk = [args_cli.n_action_steps] * N
        tasks = [""] * N
        note(f"계획 {len(plans[0])}단계" if plans[0] else "거부")
        step0 = n_steps
        limit = 3600 * max(1, max(len(p) for p in plans) // 4) + 400

        def fail(e: int, why: str):
            fail_at[e] = why
            phase[e] = FAIL
            t_done[e] = n_steps - step0
            m = one(e)
            if bool(machine.attached[e]):
                machine._detach(m)
            machine.arm_busy[m], machine.arm_hold[m], machine.arm_world[m] = False, False, False
            if e == 0:
                note(f"실패: {why}")

        def begin_back(e: int, then: int):
            m = one(e)
            machine.arm_busy[m], machine.arm_hold[m] = False, False  # 팔 관절 목표는 지금 값 그대로
            machine.back_start[m], machine.back_yaw[m] = machine.robot.data.root_pos_w[m, :2], machine.robot.data.heading_w[m]
            phase[e], after[e], t_ph[e] = BACK, then, 0
            if e == 0:
                note("뒤로 물러나기")

        for _ in range(limit):
            # ---- 1. 다음 단계 시작 ----
            for e in range(N):
                if phase[e] != NEXT:
                    continue
                if idx[e] >= len(plans[e]):
                    phase[e], t_done[e] = DONE, n_steps - step0
                    if not bool(machine.attached[e]):
                        machine.arms_to_carry(one(e))
                    if e == 0:
                        note("완료")
                    continue
                st = plans[e][idx[e]]
                skill = st.get("skill")
                m = one(e)
                t_ph[e] = 0
                if skill == "go_to" and st.get("target") in ("pick_stand", "place_stand"):
                    stage_stats["go_to"][1] += e < real
                    walk_to[e] = st["target"]
                    if st["target"] == "pick_stand" and not bool(machine.attached[e]):
                        machine.start(boxes, lay.pick_xy, lay.pick_yaw, lay.pick_xy, lay.pick_yaw, obstacles=obstacles, mask=m)
                    else:
                        xy, yaw = (lay.pick_xy, lay.pick_yaw) if st["target"] == "pick_stand" else (lay.place_xy, lay.place_yaw)
                        machine.start_walk(xy, yaw, obstacles=obstacles, keep_box=True, mask=m)
                    phase[e] = WALK
                    if e == 0:
                        note("걷기: 상자 받침대로" if st["target"] == "pick_stand" else "걷기: 판 받침대로")
                elif skill == "pick":
                    stage_stats["pick"][1] += e < real
                    color = instruction_color(st.get("instruction", ""))
                    side = next((s for s in SIDES if slot[e][s] == color), None) if color else None
                    if loc[e] != "pick_stand" or bool(machine.attached[e]):
                        fail(e, "집기: 계획 순서가 맞지 않음")
                    elif side is None:
                        fail(e, "집기: 그 색 상자가 받침대에 없음")
                    else:
                        boxes.set_color(e, color)
                        machine.box_mass[e] = group[e]["scene"]["mass"][color]
                        machine.grasp_off[e, 0, 1] = HALF + (vp.GAP_OUTER if side > 0 else vp.GAP_INNER)
                        machine.grasp_off[e, 1, 1] = -(HALF + (vp.GAP_INNER if side > 0 else vp.GAP_OUTER))
                        other.set(e, slot[e][-side], side)
                        held[e], tasks[e] = (color, side), st["instruction"]
                        lift.begin(m)
                        if args_cli.mode == "vla":
                            vexec.begin(m)
                            k_chunk[e] = args_cli.n_action_steps
                        else:
                            seq_pick.begin(m)
                        phase[e] = PICK
                        if e == 0:
                            note(f"집기: {st['instruction']}")
                elif skill == "place":
                    stage_stats["place"][1] += e < real
                    color = instruction_color(st.get("instruction", ""))
                    side = next((s for s in SIDES if pads[e][s] == color), None) if color else None
                    if loc[e] != "place_stand" or not bool(machine.attached[e]):
                        fail(e, "내려놓기: 계획 순서가 맞지 않음")
                    elif side is None:
                        fail(e, "내려놓기: 그 색 판이 없음")
                    else:
                        k = 0 if side > 0 else 1
                        place_scene.target_xy[e], place_scene.other_xy[e] = lay.pad_xy[e, k], lay.pad_xy[e, 1 - k]
                        other.set(e, None, 1)
                        held[e], tasks[e] = (held[e][0], side), st["instruction"]
                        ptrack.begin(m, place_scene)
                        if args_cli.mode == "vla":
                            vexec.begin(m)
                            k_chunk[e] = args_cli.n_action_steps
                        else:
                            seq_place.begin(m, place_scene)
                        phase[e] = PLACE
                        if e == 0:
                            note(f"내려놓기: {st['instruction']}")
                else:
                    fail(e, f"모르는 단계: {st}")
            if all(p in (DONE, FAIL) for p in phase):
                break
            # ---- 2. 팔 목표값 ----
            for e in range(N):  # 든 자세로 옮기는 동안: 팔은 관절 공간에서 든 자세로, 상자는 손 안에서 가운데로 조금씩
                if phase[e] == RECENTER:
                    a = min((t_ph[e] + 1) / RECENTER_STEPS, 1.0)
                    a = 10 * a**3 - 15 * a**4 + 6 * a**5
                    machine.ik.q_cmd[e] = rec_q[e] + a * (CARRY_Q - rec_q[e])
                    machine.att_off[e] = rec_from[e, :3] + a * (CARRY_OFF - rec_from[e, :3])
                    machine.att_yaw[e] = rec_from[e, 3] * (1.0 - a)
            arm = torch.tensor([p in (PICK, PLACE) for p in phase], device=device)
            if args_cli.mode == "vla":
                need = [e for e in range(N) if phase[e] in (PICK, PLACE) and k_chunk[e] >= args_cli.n_action_steps]
                if need:
                    img, state = vp.read_obs(u, machine)
                    ids = torch.tensor(need, device=device)
                    out = vla.infer(img[ids].cpu().numpy(), state[ids].cpu().numpy(), [tasks[e] for e in need])
                    chunk[ids] = torch.from_numpy(out).to(device)
                    for e in need:
                        k_chunk[e] = 0
                action = torch.cat([machine.ik.q_cmd, machine.attached.float()[:, None]], dim=1)
                for e in range(N):
                    if phase[e] in (PICK, PLACE):
                        action[e] = chunk[e, k_chunk[e]]
                        k_chunk[e] += 1
                vexec.pre(action, arm)
            else:
                machine.pre_step()
            # ---- 3. 한 스텝 ----
            n_steps += 1
            obs, _, _, _ = env.step(policy(obs))
            term = u.termination_manager.terminated.clone()
            cmd = machine.post_step(term)
            back = torch.tensor([p == BACK for p in phase], device=device)
            back_cmd, back_done = machine._backoff_command()
            cmd[back] = back_cmd[back]
            cmd_term.vel_command_b[:] = cmd
            obs = env.get_observations()
            if args_cli.video:
                head_frames.append(u.scene.sensors["vla_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().copy())
            if n_steps % 400 == 0:
                mem["max"] = max(mem["max"], gpu_used_mb())
            # ---- 4. 팔 쪽 갱신 ----
            picking = torch.tensor([p == PICK for p in phase], device=device)
            placing = torch.tensor([p == PLACE for p in phase], device=device)
            if args_cli.mode == "vla":
                vexec.post(action, arm, term, other=other)
            else:
                seq_pick.fail(term & seq_pick.active)
                seq_pick.update()
                seq_place.fail(term & seq_place.active)
                seq_place.update()
            lift.update(picking & ~term)
            ptrack.update(placing & ~term)
            # ---- 5. 단계 전환 ----
            for e in range(N):
                ph = phase[e]
                if ph in (NEXT, DONE, FAIL):
                    continue
                if bool(term[e]):
                    machine.ik.reset(torch.tensor([e], device=device))
                    fail(e, "넘어짐")
                    continue
                t_ph[e] += 1
                if ph == WALK:
                    state_e = int(machine.state[e])
                    if state_e == ppm.S_DONE:
                        stage_stats["go_to"][0] += e < real
                        loc[e] = walk_to[e]
                        idx[e] += 1
                        phase[e] = NEXT
                    elif state_e == ppm.S_FAIL:
                        fail(e, "걷기: " + ppm.STATE_NAMES[int(machine.fail_state[e])])
                elif ph == PICK:
                    if args_cli.mode == "vla":
                        done_ok = bool(lift.success[e])
                        bad = "다른 상자를 집음" if bool(vexec.wrong_pick[e]) else (vp.classify_failure(vexec, lift, e) if t_ph[e] > T_ARM else None)
                    else:
                        fin = int(seq_pick.phase[e])
                        done_ok = fin == seq_pick.DONE and bool(lift.success[e])
                        bad = None if done_ok or fin not in (seq_pick.DONE, seq_pick.FAIL) else "손이 잡는 위치에 들어가지 못함"
                    if done_ok:
                        stage_stats["pick"][0] += e < real
                        slot[e][held[e][1]] = None
                        begin_back(e, RECENTER)
                    elif bad:
                        fail(e, "집기: " + bad)
                elif ph == BACK:
                    if bool(back_done[e]):
                        phase[e], t_ph[e] = WAIT, 0
                elif ph == WAIT:
                    if t_ph[e] >= WAIT_STEPS:
                        if after[e] == RECENTER:  # 양손을 든 자세로 옮기며 상자를 손 가운데로
                            rec_from[e, :3], rec_from[e, 3] = machine.att_off[e], machine.att_yaw[e]
                            rec_q[e] = machine.ik.q_cmd[e]
                            machine.att_floor[one(e)] = -1.0e9
                            phase[e], t_ph[e] = RECENTER, 0
                            if e == 0:
                                note("든 자세로")
                        else:  # 내려놓은 뒤: 팔을 기본 자세로 되돌림 (다음 집기가 시범과 같은 자세에서 시작하게)
                            machine.arms_to_carry(one(e))
                            phase[e], t_ph[e] = HOME, 0
                            if e == 0:
                                note("팔 내리기")
                elif ph == HOME:
                    if not bool(machine.j_move[e]) and t_ph[e] > 5:
                        idx[e] += 1
                        phase[e] = NEXT
                elif ph == RECENTER:
                    if t_ph[e] >= RECENTER_STEPS + 10:
                        machine.att_off[e], machine.att_yaw[e] = CARRY_OFF, 0.0
                        machine.grasp_off[e, 0, 1], machine.grasp_off[e, 1, 1] = HALF + vpl.CARRY_GAP, -(HALF + vpl.CARRY_GAP)
                        machine.arm_hold[one(e)] = False  # 걷는 동안은 팔 관절 목표 고정
                        idx[e] += 1
                        phase[e] = NEXT
                elif ph == PLACE:
                    if args_cli.mode == "vla":
                        done_ok = bool(ptrack.success[e])
                        late = bool(ptrack.released[e]) and int(ptrack.t[e] - ptrack.t_release[e]) > GIVE_UP_STEPS
                        bad = None if done_ok or not (late or t_ph[e] > T_ARM) else ptrack.outcome(e)[0]
                    else:
                        fin = int(seq_place.phase[e])
                        done_ok = fin == seq_place.DONE and bool(ptrack.success[e])
                        bad = None if done_ok or fin not in (seq_place.DONE, seq_place.FAIL) else ptrack.outcome(e)[0]
                    if done_ok:
                        stage_stats["place"][0] += e < real
                        pad_box[e][held[e][1]] = held[e][0]
                        held[e] = None
                        begin_back(e, NEXT)
                    elif bad:
                        fail(e, "내려놓기: " + bad)
        for e in range(N):
            if phase[e] not in (DONE, FAIL):
                fail(e, "시간 초과")
        for _ in range(80 if args_cli.video else 10):  # 마무리 (팔을 내리고 상자가 자리 잡게)
            machine.pre_step()
            n_steps += 1
            obs, _, _, _ = env.step(policy(obs))
            cmd_term.vel_command_b[:] = machine.post_step(u.termination_manager.terminated.clone())
            obs = env.get_observations()
            if args_cli.video:
                head_frames.append(u.scene.sensors["vla_cam"].data.output["rgb"][0, ..., :3].cpu().numpy().copy())
        # ---- 결과 ----
        for e in range(real):
            sc = group[e]
            expected = normalize(json.loads(sc["expected"]))
            if sc["kind"] == "refuse":
                ok = preds[e] is not None and preds[e]["error"] == expected["error"] and not preds[e]["plan"]
                final_ok = None
            else:
                final_ok = True
                for box_c, pad_c in sc["pairs"]:  # 옮기라고 한 상자가 그 색 판 위에 바로 놓여 있는지
                    side = next(s for s in SIDES if sc["scene"]["pads"][s] == pad_c)
                    box = u.scene[f"small_{box_c}_box"]
                    pos = box.data.root_pos_w[e]
                    on = float(torch.norm(pos[:2] - lay.pad_xy[e, 0 if side > 0 else 1])) <= vpl.PLACE_OK_XY \
                        and abs(float(pos[2]) - vpl.BOX_REST_Z) <= 0.03 and float(vpl.box_tilt_deg(box.data.root_quat_w)[e]) <= vpl.PLACE_OK_TILT
                    final_ok = final_ok and on
                ok = phase[e] == DONE and final_ok
            records.append({
                "id": sc["id"], "kind": sc["kind"], "template": sc["template"], "outcome": sc["outcome"], "command": sc["command"],
                "expected": sc["expected"], "vlm_raw": plans_cache[sc["id"]]["raw"], "vlm_ok": bool(vlm_ok[e]),
                "vlm_latency_s": plans_cache[sc["id"]]["latency_s"], "success": bool(ok), "fail_at": fail_at[e],
                "steps_done": idx[e], "steps_total": len(plans[e]), "final_ok": final_ok,
                "time_s": round(t_done[e] * step_dt, 1) if t_done[e] is not None else None,
            })
        done_n = sum(r["success"] for r in records)
        print(f"[full] 묶음 {r_i + 1}/{len(rounds)} ({group[0]['kind']}): 누적 성공 {done_n}/{len(records)}, {time.time() - t_start:.0f}초", flush=True)

if vla is not None:
    vla_latency = [v[0] for v in vla.latencies]
    vla.close()
if vlm is not None:
    vlm.close()
if args_cli.plans_out:
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.plans_out)), exist_ok=True)
    with open(args_cli.plans_out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in plans_cache.items()}, f, ensure_ascii=False, indent=1)

# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
label = f"VLA (행동 묶음 중 {args_cli.n_action_steps}스텝 실행)" if args_cli.mode == "vla" else "스크립트 IK"
print(f"\n========== 전체 연결: 팔 = {label}, 시나리오 {len(records)}개 ==========")
summary = {}
for kind, name in (("one", "상자 하나 옮기기"), ("two", "상자 둘 옮기기"), ("refuse", "거부해야 하는 명령")):
    rows = [r for r in records if r["kind"] == kind]
    if not rows:
        continue
    ok = sum(r["success"] for r in rows)
    plan_ok = sum(r["vlm_ok"] for r in rows)
    summary[kind] = {"n": len(rows), "success": ok, "vlm_ok": plan_ok}
    line = f"{name}: 성공 {ok}/{len(rows)} ({ok / len(rows):.0%}) | VLM 계획이 정답과 같음 {plan_ok}/{len(rows)}"
    times = [r["time_s"] for r in rows if r["success"] and r["time_s"]]
    if kind != "refuse" and times:
        line += f" | 걸린 시간 (성공한 것, 초): 평균 {np.mean(times):.1f}, 최대 {np.max(times):.1f}"
    print(line)
    fails = {}
    for r in rows:
        if not r["success"]:
            key = r["fail_at"] or ("VLM 답이 틀림" if not r["vlm_ok"] else "끝까지 했지만 상자가 판 위에 없음")
            fails[key] = fails.get(key, 0) + 1
    if fails:
        print("  실패한 곳: " + ", ".join(f"{k} {v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])))
print("단계별 성공 (성공/시도): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in stage_stats.items())
      + f" | VLM 계획 {sum(r['vlm_ok'] for r in records)}/{len(records)}")
falls = sum(1 for r in records if r["fail_at"] == "넘어짐")
print(f"넘어짐: {falls}")
lat = [r["vlm_latency_s"] for r in records]
print(f"VLM 생성 시간 (초): 평균 {np.mean(lat):.2f}, 최대 {np.max(lat):.2f}" + (" (저장해 둔 답을 다시 씀)" if args_cli.plans_from else ""))
if vla_latency:
    print(f"VLA 응답 시간 (초, 요청 한 번): 평균 {np.mean(vla_latency):.3f}, 최대 {np.max(vla_latency):.3f}, 요청 {len(vla_latency)}번")
print("GPU 전체 사용량 (MB): " + ", ".join(f"{k} {v}" for k, v in mem.items()))
print(f"전체 걸린 시간 {time.time() - t_start:.0f}초")
if args_cli.out:
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"mode": args_cli.mode, "n_action_steps": args_cli.n_action_steps, "seed": args_cli.seed, "summary": summary,
                   "stage_stats": stage_stats, "gpu_mb": mem, "vla_latency_mean_s": float(np.mean(vla_latency)) if vla_latency else None,
                   "records": records}, f, ensure_ascii=False, indent=1)
    print(f"결과 저장: {args_cli.out}")

if args_cli.video:
    r0 = records[0]
    info = {"name": args_cli.name, "label": label, "command": r0["command"], "plan": r0["vlm_raw"], "success": r0["success"],
            "fps": int(round(1 / step_dt)), "total_steps": n_steps, "events": events}
    with open(os.path.join(args_cli.video_dir, f"{args_cli.name}_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    iio.imwrite(os.path.join(args_cli.video_dir, f"{args_cli.name}_head.mp4"), np.stack(head_frames), fps=info["fps"])
    print(f"[full] 영상: {os.path.join(args_cli.video_dir, args_cli.name + '-step-0.mp4')}")

env.close()
simulation_app.close()
