# humanoid_project

Unitree G1 휴머노이드가 한국어 명령을 듣고 상자를 집어 옮기도록 만드는 프로젝트입니다. NVIDIA Isaac Lab 시뮬레이터에서 진행합니다.

```
머리 카메라 사진 + 한국어 명령
        │  소형 VLM (LoRA 파인튜닝)
        ▼
기술 순서 JSON   {"plan": [{"skill": "pick", ...}, {"skill": "place", ...}]}
        │  스크립트 (상태 머신, 양팔 IK)
        ▼
걷기·균형 = 강화학습 정책 (PPO),  팔 = IK,  잡기 = 상자를 손에 고정
```

## 단계

| 단계 | 내용 | 방법 |
|---|---|---|
| 1~2 | 걷기, 짐 든 채 걷기 | 강화학습 (속도 명령 따라가기, 손 무게 0~5kg) |
| 3 | 사람에 가까운 걸음 | 걸음 박자·발 높이·미끄러짐 보상 |
| 4 | 팔이 움직여도 버티기 | 팔 목표 자세 무작위화 |
| 5 | 상자 집어 옮기기 | 양팔 IK + 상태 머신 (학습 없음) |
| 6 | 걷기 정책 v2 | 팔 전체 무작위화, 뒤로 걷기, 어느 박자에서든 멈추기 |
| 7 | VLM 판단 층 | 사진 + 한국어 명령 → 기술 순서 JSON |
| 8 | VLA 상체 제어 | 머리 카메라 사진 + 팔 상태 + 한국어 명령 → 팔 관절 목표값과 잡기 신호 (SmolVLA 미세조정) |
| 9 | 내려놓기도 VLA 로 | 집기와 내려놓기를 한 모델로: 명령한 색의 상자를 집고, 명령한 색의 판 위에 내려놓기 (전체 연결은 아직) |

단계별 설계 결정, 결과 표, 시행착오는 [CLAUDE.md](CLAUDE.md) 에 있습니다.

## 폴더

- `humanoid_tasks/g1_carry/` — 과제 설정 (환경, 보상·이벤트, IK, 상자 집어 옮기기 상태 머신, VLM 장면)
- `scripts/` — 학습·평가·녹화·데이터 생성·데모 실행
- `vlm/` — VLM 판단 층 (명령 체계, 문장 템플릿, 파인튜닝, 평가, 실행용 작업자)
- `vla/` — VLA 상체 제어 (시범 데이터 변환·합치기, SmolVLA 미세조정, 행동 출력 작업자, 명령 바꿔치기 시험, 메모리 측정)

## 실행

Isaac Lab 2.3.2 (Isaac Sim 5.1) 환경에서 프로젝트 폴더를 기준으로 실행합니다. 설치 순서는 CLAUDE.md 에 있습니다.

```
python scripts\train.py --task Humanoid-G1-Loco-V2-Flat-v0 --headless --num_envs 4096 --max_iterations 4000
python scripts\eval_carry.py --headless --task Humanoid-G1-Loco-V2-Flat-Play-v0 --load_run ".*_loco_v2"
python scripts\pick_place_demo.py --num_envs 100 --randomize --headless
```

VLM 판단 층 (데이터 생성은 Isaac 환경에서, 파인튜닝·평가는 별도 파인튜닝 환경에서):

```
python scripts\make_vlm_data.py --headless
cd vlm
python evaluate.py --model Qwen/Qwen3.5-0.8B --name base_08b
python train.py
python evaluate.py --model Qwen/Qwen3.5-0.8B --adapter outputs/qwen35_08b/lora --name ft_08b
```

VLA 상체 제어 (시범 수집·실행은 Isaac 환경에서, 변환·학습은 LeRobot 을 설치한 별도 환경에서):

```
python scripts\collect_vla_demos.py --headless --episodes 208 --num_envs 16 --out data\vla_pick\raw_v1
python vla\convert_to_lerobot.py --raw data\vla_pick\raw_v1 --root data\vla_pick\lerobot_v1
python vla\train.py --dataset_root data\vla_pick\lerobot_v1 --steps 8000 --batch 32 --out data\vla_pick\train\main_8k
python scripts\vla_pick_eval.py --headless --mode vla --ckpt data\vla_pick\train\main_8k\checkpoints\last\pretrained_model
```

색이 다른 상자 두 개 중 명령한 색을 집기 (같은 장면에 명령만 다른 쌍으로 시범을 모음):

```
python scripts\collect_vla_demos.py --two_box --headless --episodes 408 --num_envs 24 --stance_y 0,0.06 --stance_yaw 0,8 --out data\vla_pick\raw3_v1
python vla\convert_to_lerobot.py --raw data\vla_pick\raw3_v1 --root data\vla_pick\lerobot3x_ko --repo_id local/g1_vla_pick3x_ko --cross_frames 56
python vla\train.py --dataset_root data\vla_pick\lerobot3x_ko --repo_id local/g1_vla_pick3x_ko --steps 8000 --batch 32 --train_vlm --out data\vla_pick\train\cross_ko_vlm_8k
python vla\swap_test.py --ckpt data\vla_pick\train\cross_ko_vlm_8k\checkpoints\last\pretrained_model --raw data\vla_pick\raw3_val
python scripts\vla_pick_eval.py --headless --two_box --pairs --mode vla --n_action_steps 50 --episodes 96 --num_envs 24 --ckpt data\vla_pick\train\cross_ko_vlm_8k\checkpoints\last\pretrained_model
```

든 상자를 명령한 색의 판 위에 내려놓기, 집기와 한 모델로 (위의 집기 데이터에 내려놓기 시범을 합쳐 이어서 학습. 실제로 쓴 명령 전체는 CLAUDE.md 9단계):

```
python scripts\collect_vla_place.py --headless --episodes 440 --num_envs 24 --n_single 6 --stance_y 0,0.10 --stance_yaw 0,10 --out data\vla_place\raw_v1
python vla\convert_to_lerobot.py --raw data\vla_place\raw_v1 --root data\vla_place\lerobot_v1 --repo_id local/g1_vla_place --cross_frames 54
python vla\merge_datasets.py --roots data\vla_pick\lerobot3x_ko data\vla_place\lerobot_v1 --repo_ids local/g1_vla_pick3x_ko local/g1_vla_place --out data\vla_place\lerobot_joint
python vla\train.py --dataset_root data\vla_place\lerobot_joint --repo_id local/g1_vla_joint --policy data\vla_pick\train\cross_ko_vlm_8k\checkpoints\last\pretrained_model --steps 14000 --batch 32 --train_vlm --out data\vla_place\train\joint_14k
python scripts\vla_place_eval.py --headless --mode vla --episodes 96 --num_envs 24 --n_single 6 --ckpt data\vla_place\train\joint_14k\checkpoints\last\pretrained_model
```

## 모델과 데이터

- **VLM**: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 를 LoRA 로 파인튜닝해 씁니다. 모델 카드에 표기된 라이선스는 Apache License 2.0 입니다. 모델 가중치와 학습한 어댑터는 이 저장소에 들어 있지 않습니다.
- **VLM 학습 데이터**: 모두 이 프로젝트에서 만든 합성 데이터입니다. 사진은 Isaac Lab 장면(받침대 3개, 색 상자)을 로봇 머리 카메라로 찍은 것이고, 명령 문장과 정답 JSON 은 `vlm/templates.py` 의 템플릿이 장면의 실제 상태로부터 자동으로 만듭니다. 외부 데이터셋은 쓰지 않았습니다. 사진은 용량이 커서 저장소에 올리지 않고, `scripts/make_vlm_data.py` 를 같은 시드로 실행하면 같은 장면·문장으로 다시 만들 수 있습니다.
- **VLA**: [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (`lerobot/smolvla_base`, 4.5억 파라미터)를 [LeRobot](https://github.com/huggingface/lerobot) 으로 미세조정해 씁니다. 모델 카드에 표기된 라이선스는 Apache License 2.0 이고, 바탕이 되는 VLM(SmolVLM2-500M-Video-Instruct)도 Apache License 2.0 입니다. LeRobot 코드도 Apache License 2.0 입니다. 가중치와 학습한 체크포인트는 이 저장소에 들어 있지 않습니다.
- **VLA 시범 데이터**: 이 프로젝트의 상자 집기 스크립트(상태 머신 + IK)를 Isaac Lab 에서 돌려 자동으로 모은 합성 데이터입니다 (머리 카메라 영상, 팔 관절 위치·목표값, 잡기 신호, 한국어 명령. 집기와 내려놓기). 외부 데이터셋은 쓰지 않았고, 용량이 커서 저장소에 올리지 않습니다.
- **로봇 모델**: Isaac Lab 에 들어 있는 Unitree G1 설정을 상속해 씁니다.
