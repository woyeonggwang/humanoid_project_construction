"""명령 바꿔치기 시험 (Isaac 없이): 모델이 명령 문장의 색을 쓰는지 본다.

시범 원자료(ep_*.npz, scripts/collect_vla_demos.py --two_box)의 한 시점 관측을 넣고, 같은 잡음에서 명령의 색만 옆 상자의 색으로 바꿔 행동 묶음을 비교한다.
- 시점은 손을 위로 올리는 구간의 끝 (기본 45스텝째): 아직 좌우로 갈라지기 전이라 사진·상태만으로는 어느 상자로 갈지 알 수 없다
- 어느 쪽으로 가는지 = 행동 묶음 끝에서 양쪽 어깨 roll 목표값의 합이 지금보다 커지는지 작아지는지 (시범의 부호와 비교)
- 명령을 쓰는 모델: 맞는 명령이면 시범과 같은 쪽, 바꾼 명령이면 반대쪽으로 간다. 명령을 쓰지 않는 모델: 바꿔도 같은 쪽으로 간다

LeRobot 을 설치한 별도 환경에서 실행한다 (Isaac 환경이 아님).
예) python vla/swap_test.py --ckpt <체크포인트>/pretrained_model --raw data/vla_pick/raw3_val
"""

import argparse
import glob
import json

import numpy as np
import torch

# 색 낱말 (humanoid_tasks/g1_carry/vla_pick.py 의 COLOR_WORDS 와 같음. 그 파일은 Isaac 을 import 하므로 여기서 불러올 수 없음)
COLOR_WORDS = {"red": ["빨간색", "빨간"], "blue": ["파란색", "파란"], "green": ["초록색", "초록"], "yellow": ["노란색", "노란"]}


def swap_color(task: str, color: str, other: str, lang: str) -> str:
    """명령 문장에서 색 낱말만 other 의 것으로 바꾼다 (낱말 꼴은 그대로: '빨간색' ↔ '파란색', '빨간' ↔ '파란')."""
    if lang == "en":
        assert color in task, task
        return task.replace(color, other)
    for k, word in enumerate(COLOR_WORDS[color]):  # 긴 꼴부터 찾음
        if word in task:
            return task.replace(word, COLOR_WORDS[other][k])
    raise ValueError(f"색 낱말을 찾지 못함: {task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="체크포인트의 pretrained_model 폴더")
    ap.add_argument("--raw", required=True, help="시범 원자료 폴더 (학습에 쓰지 않은 것을 권장)")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--n", type=int, default=48, help="쓸 에피소드 수")
    ap.add_argument("--step", type=int, default=45, help="관측을 뽑을 시점 (스텝)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(args.ckpt).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
    files = sorted(glob.glob(args.raw + "/ep_*.npz"))
    pick = np.random.default_rng(args.seed).choice(len(files), min(args.n, len(files)), replace=False)

    def run(img, state, task, seed):
        torch.manual_seed(seed)
        batch = {"observation.images.head": torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0,
                 "observation.state": torch.from_numpy(state)[None].float(), "task": [task]}
        with torch.inference_mode():
            return post(policy.predict_action_chunk(pre(batch)))[0].float().cpu().numpy()

    d_noise, d_swap, same_true, opp_swap = [], [], [], []
    t0 = args.step
    for i in pick:
        d = np.load(files[i])
        meta = json.loads(str(d["meta"]))
        img, state, demo = d["images"][t0], d["state"][t0], d["action"][t0:t0 + 50]
        task = meta["task_en"] if args.lang == "en" else meta["task"]
        swapped = swap_color(task, meta["color"], meta["other_color"], args.lang)
        a_true, a_noise, a_swap = run(img, state, task, 1), run(img, state, task, 2), run(img, state, swapped, 1)
        d_noise.append(np.abs(a_true[:, :10] - a_noise[:, :10]).mean())
        d_swap.append(np.abs(a_true[:, :10] - a_swap[:, :10]).mean())
        side = lambda a: np.sign((a[-1, 1] + a[-1, 6]) - (state[1] + state[6]))  # noqa: E731
        same_true.append(side(a_true) == side(demo))
        opp_swap.append(side(a_swap) == -side(demo))
    print(f"명령 바꿔치기 시험: 에피소드 {len(pick)}개, {t0}스텝째 관측, 명령 {'영어' if args.lang == 'en' else '한국어'}")
    print(f"맞는 명령일 때 시범과 같은 쪽으로 감: {np.mean(same_true):.0%}")
    print(f"명령의 색만 바꿨을 때 반대쪽으로 감: {np.mean(opp_swap):.0%}  (명령을 쓰면 높고, 쓰지 않으면 낮음)")
    print(f"행동 차이 (팔 목표값 rad 평균): 명령만 바꿈 {np.mean(d_swap):.4f}, 잡음만 바꿈 {np.mean(d_noise):.4f}")


if __name__ == "__main__":
    main()
