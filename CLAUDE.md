# humanoid_project — CLAUDE.md

휴머노이드 로봇 보행을 강화학습으로 학습하는 프로젝트. warehouse-robot-llm(한국어 명령 → 소형 LLM → 강화학습 바퀴 로봇) 다음 단계.

- 진행 계획 (2026-11-17 논의, 확정 아님)
  1. 걷기: 속도 명령(앞뒤·옆·회전)을 따라 걷는 정책을 강화학습으로 학습
  2. 기존 시스템과 연결: warehouse-robot-llm 의 계획 층(소형 LLM·점수 모델·9B 판단)을 바퀴 로봇 대신 휴머노이드에 연결
  3. VLA: 물건 집기 같은 조작은 이후 VLA 로 연동. 이때도 걷기·균형 같은 움직임 제어는 강화학습 정책이 맡는 구조 예상 (모델별 구조는 확인 필요)

## 작업 규칙 (Claude)
- 버전·환경 고정 사항, 확정된 설계 결정과 이유, 시행착오(증상·원인·해결), 미해결 과제는 이 파일에 기록한다. 확인된 것만 적고, 확인되지 않은 것은 "(확인 필요)"로 표시한다.
- 백업 파일은 사용자가 요청할 때만 만든다.
- 사용자 PC 의 환경·설치된 프로그램·모델 같은 사적인 정보는 저장소에 들어가는 문서·코드·커밋 메시지에 적지 않는다. 쓰지 않은 대안은 기록하지 않고, 사용하는 것과 그 기술적 이유만 적는다.
- Isaac Lab 원본 소스는 수정하지 않는다. 바꿀 내용은 이 프로젝트 안에서 원본 설정을 상속해 만든다.

## 저장소 구조 (2026-11-19 결정)
```
humanoid_project/
├─ humanoid_tasks/g1_carry/   과제: __init__.py(gym 등록), env_cfg.py(환경), agents.py(PPO)
├─ scripts/                   _launch.py(공통), train.py, play.py, inspect_g1.py(점검)
├─ logs/                      학습 기록 (git 제외)
├─ outputs/                   Hydra 실행 기록 (git 제외)
└─ portfolio/                 (git 제외)
```
- 실행은 프로젝트 폴더에서: Isaac Lab conda 환경을 켜고 프로젝트 폴더로 이동한 뒤 `python scripts\train.py ...` (학습 기록이 실행 위치 기준 `logs\rsl_rl\<experiment_name>\` 에 생김 — 원본 train.py·play.py 코드로 확인)
- 과제 등록 방식: `scripts/_launch.py` 가 프로젝트 루트를 sys.path 에 넣고 `humanoid_tasks.g1_carry` 를 import(= gym.register) 한 뒤, Isaac Lab 원본 `scripts/reinforcement_learning/rsl_rl/train.py`(또는 play.py)를 runpy 로 실행. 명령줄 인자는 그대로 전달
  - 원본 스크립트가 `import cli_args` 를 하므로 그 폴더도 sys.path 에 추가
  - Isaac Lab 위치는 환경 변수 ISAACLAB_PATH, 없으면 설치된 isaaclab 패키지 위치에서 계산
  - gym.register 는 Isaac Sim 앱이 켜지기 전에 실행되므로 설정 위치를 문자열로만 넘기고 isaaclab 모듈은 import 하지 않음 (앱 전 import 시 오류 여부는 확인 필요)
- 원격 저장소: https://github.com/woyeonggwang/humanoid_project_construction (2026-11-19 구성, `main` 브랜치). 커밋 전 개인정보(로컬 경로·PC 사양·이메일) 검사
- .gitignore: `portfolio/`, `logs/`, `outputs/`, `__pycache__/`, `*.pt`
  - `outputs/`: train.py·play.py 실행 때 Hydra 가 실행 위치에 만드는 실행 기록. hydra.yaml 에 실행 폴더의 절대 경로가 들어가므로 올리지 않음
  - `logs/`: 체크포인트·영상이 커서 올리지 않음
- `portfolio/`: 블로그·포트폴리오용 결과물(GIF·영상). **git 에 올리지 않음**
  - `portfolio/g1_walk_play.gif`: 1차 본 학습(model_1499) play 영상을 GIF 로 변환 (800px, 15fps, 약 10초, 약 5.8MB). 원본 mp4 는 학습 폴더의 `videos\play\`

## 확정된 설계 결정
- 시뮬레이터: NVIDIA Isaac Lab (2026-11-17 사용자 결정)
  - 이유: 휴머노이드 강화학습 연구에서 많이 쓰는 도구이고 Unitree G1 보행 과제(Isaac-Velocity-Flat-G1-v0 등)가 들어 있음. Isaac Lab 2.3 은 Windows 11 을 공식 지원. 공식 성능 자료상 Isaac-Velocity-Rough-G1 을 환경 4096개로 돌릴 때 VRAM 6.1GB·RAM 6.5GB (Ubuntu, headless) → 화면 없이(headless) 학습하면 일반 데스크톱 GPU 로도 가능하다고 판단
- 로봇: Unitree G1 (Isaac Lab 내장 과제 사용)

## 환경 (2026-11-19 설치·확인)
- OS: Windows 11 (명령은 cmd 기준)
- Windows 긴 경로 허용 켬 (레지스트리 LongPathsEnabled = 1)
- conda 환경: Isaac Lab 전용 환경을 따로 만들어 사용 (아래 설치 순서)
- Isaac Lab 소스: `git clone --branch v2.3.2` 로 받은 폴더 (이하 `<IsaacLab>`)
- 주요 버전 (확인함)
  - Python 3.11.18
  - isaacsim 5.1.0 (pip, `isaacsim[all,extscache]==5.1.0`)
  - torch 2.7.0+cu128, torchvision 0.22.0+cu128, torchaudio 2.7.0+cu128 (GPU 인식 True)
  - isaaclab 0.54.2, rsl-rl-lib 3.1.2, tensordict 0.8.3, flatdict 4.0.1
  - setuptools 79.0.1, packaging 23.0
- 남아 있는 의존성 충돌 경고 (실행 영향 확인 필요): starlette 0.49.1 (isaaclab 이 고정) vs fastapi 0.115.7 의 starlette<0.46 요구, psutil 7.2.2 vs isaacsim-kernel 의 5.9.8 요구, typing_extensions 4.16.0 vs isaacsim-kernel 의 4.12.2 요구
- 사용자 폴더(설치 드라이브와 무관)에 생기는 것: Isaac Sim 캐시 `%LOCALAPPDATA%\ov\cache`, pip 빌드 임시 폴더 `%TEMP%`, Isaac Lab 로그 `%TEMP%\isaaclab\logs`
- pip 캐시 위치를 바꾸려면 `set PIP_CACHE_DIR=<폴더>` (창마다 다시 지정해야 함)

## 설치 순서 (실제로 성공한 순서)
명령 프롬프트(cmd) 기준. PowerShell 이면 환경 변수는 `$env:PIP_CACHE_DIR = "..."` 형식. `<ENV>` 는 conda 환경 경로, `<DIR>` 은 Isaac Lab 을 받을 폴더.
```
conda create -p <ENV> python=3.11
conda activate <ENV>
set PIP_CACHE_DIR=<폴더>   (선택)
python -m pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install --force-reinstall --no-deps torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
cd /d <DIR>
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
isaaclab.bat --install rsl_rl
pip install "setuptools<80"
pip install flatdict==4.0.1 --no-build-isolation
pip install -e source\isaaclab
pip install --no-deps packaging==23.0
pip install "tensordict>=0.8,<0.9"
pip install --force-reinstall --no-deps torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```
확인: `python -c "import torch, isaaclab, rsl_rl, tensordict; print(torch.__version__, torch.cuda.is_available(), tensordict.__version__)"` → `2.7.0+cu128 True 0.8.3`

## 실행
- 학습 (화면 없이): `isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task=Isaac-Velocity-Flat-G1-v0 --headless --num_envs 1024 --max_iterations 50`
- 학습 기록: `<IsaacLab>\logs\rsl_rl\g1_flat\<날짜시간>` (Isaac Lab 폴더에서 실행했을 때)
- 결과 영상 녹화 (play, 2026-11-19 동작 확인): `isaaclab.bat` 를 거치지 않고 python 으로 직접 실행하며 D3D12 렌더러를 지정 (이유는 아래 시행착오)
  ```
  python scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16 --headless --video --video_length 500 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
  ```
  - 가장 최근 학습 폴더의 마지막 체크포인트를 자동으로 불러옴 (이번에는 `model_1499.pt`)
  - 영상: `<학습 폴더>\videos\play\rl-video-step-0.mp4` (1280x720, 50fps, 500스텝 = 약 10초)
  - 두 kit 인자 중 어느 것이 실제로 효과를 냈는지는 따로 확인하지 않음 (확인 필요)

## 학습 결과
### 설치 확인용 시험 (2026-11-19)
- Isaac-Velocity-Flat-G1-v0, 환경 1024개, 50회 반복, headless → 50.8초, 오류 없음
- 속도 약 2.5만~2.9만 스텝/초, VRAM 약 4GB (사용자가 작업 관리자로 확인)
- 과제 구성: 동작(관절 위치 목표) 37, 관측 123 (기체 선속도·각속도, 중력 방향, 속도 명령, 관절 위치·속도, 이전 동작), 보상 항목 16개, 종료 조건 time_out·base_contact(넘어짐)
- 평균 보상 -6.18 → -5.40, 에피소드 길이 11 → 47. base_contact 1.0 = 모든 에피소드가 넘어져서 끝남 (50회로는 걷기를 배우기에 짧음, 동작 확인용)

### 본 학습 1차 (2026-11-19)
- Isaac-Velocity-Flat-G1-v0, 환경 4096개, 1500회 반복(과제 기본값, 확인함), headless → 1868초(약 31분), 총 1억 4746만 스텝
- 속도 약 7.8만~8.4만 스텝/초, 반복 1회 약 1.2초. 전용 GPU 메모리 약 5.7GB(다른 프로그램 포함), GPU 사용률 약 70% (작업 관리자, 사용자 확인)
- 학습 기록 폴더: `<IsaacLab>\logs\rsl_rl\g1_flat\2026-11-19_15-11-59` (체크포인트 `model_1499.pt`)
- 지표 변화
  | 지표 | 90회 | 624회 | 1031회 | 1499회(마지막) |
  |---|---|---|---|---|
  | 평균 보상 | -5.96 | 26.08 | 27.75 | 28.05 |
  | 평균 에피소드 길이 (최대 1000) | 131 | 1000 | 1000 | 1000 |
  | 이동 속도 따라가기 보상 | 0.035 | 0.916 | 0.922 | 0.925 |
  | 회전 속도 따라가기 보상 | 0.012 | 0.738 | 0.781 | 0.778 |
  | 이동 속도 오차 | 0.22 | 0.26 | 0.25 | 0.245 |
  | 회전 속도 오차 | 0.86 | 0.52 | 0.45 | 0.46 |
  | 넘어짐(base_contact) | 1.0 | 0.0032 | 0.0017 | 0.0042 |
  | 동작 노이즈 std | 1.0 | 0.65 | 0.60 | 0.60 |
- 해석: 넘어지지 않고(시간 종료 99.6%) 명령 속도를 따라 걸음. 600회 무렵 이후는 조금씩 다듬는 단계. 90회 때의 속도 오차는 에피소드가 짧을 때라 이후 값과 바로 비교하기 어려움
- play 영상 (2026-11-19, D3D12 렌더러): 16대가 넘어지지 않고 서서 걸어 다니는 것을 영상 프레임으로 확인. 영상은 `videos\play\rl-video-step-0.mp4`

## 시행착오 기록
### 2026-11-19 pip 업그레이드가 base 환경에 됨
- 증상: `python -m pip install --upgrade pip` 결과가 conda 기본(base) 환경에 설치됨, 프롬프트가 `(base)`
- 원인: `conda activate <ENV>` 를 하지 않은 상태
- 해결: 환경을 켠 뒤 프롬프트가 `(<ENV>)` 인지 확인하고 진행

### 2026-11-19 `$env:PIP_CACHE_DIR` 가 오류
- 증상: "파일 이름, 디렉터리 이름 또는 볼륨 레이블 구문이 잘못되었습니다"
- 원인: 창이 PowerShell 이 아니라 명령 프롬프트(cmd)
- 해결: cmd 에서는 `set PIP_CACHE_DIR=<폴더>`

### 2026-11-19 Isaac Sim 설치가 CPU 용 torch 를 깖
- 증상: isaacsim 설치 기록에 `torch-2.7.0-cp311-cp311-win_amd64.whl` (PyPI)
- 원인: Windows 용 PyPI torch 는 CPU 전용
- 해결: cu128 저장소에서 `--force-reinstall --no-deps` 로 torch·torchvision·torchaudio 교체. 같은 버전 번호(2.7.0)가 이미 있으면 일반 설치는 건너뛰므로 강제 재설치 필요

### 2026-11-19 isaaclab 설치 실패: flatdict 빌드 오류
- 증상: `isaaclab.bat --install rsl_rl` 후 `ModuleNotFoundError: No module named 'isaaclab'`. 설치 기록에 `Failed to build 'flatdict'`, `No module named 'pkg_resources'`
- 원인: isaaclab 이 flatdict==4.0.1 을 고정. 4.0.1 은 소스 배포판이라 빌드할 때 pkg_resources 가 필요한데, 설치된 setuptools 84 에서는 pkg_resources 를 쓸 수 없어 빌드 실패 (setuptools 의 pkg_resources 제거 시점은 확인 필요). flatdict 4.1.0 을 따로 넣어도 isaaclab 이 4.0.1 을 다시 요구해 같은 오류
- 해결: `pip install "setuptools<80"` → `pip install flatdict==4.0.1 --no-build-isolation` → `pip install -e source\isaaclab`

### 2026-11-19 rsl_rl 설치가 torch 를 2.15(CPU)로 올림
- 증상: 설치 중 tensordict 0.15.6 이 깔리면서 torch 2.15.0·torchvision 0.29.2 (PyPI, CPU) 로 교체. isaaclab.bat 이 torch 2.7.0+cu128 로 되돌렸지만 `tensordict 0.15.6 requires torch>=2.11` 충돌과 torchaudio 삭제가 남음
- 원인: tensordict 최신판이 torch 2.11 이상을 요구
- 해결: `pip install "tensordict>=0.8,<0.9"` (0.8.3, torch 를 바꾸지 않음) + torchaudio cu128 재설치

### 2026-11-19 pytest 가 packaging 을 26.3 으로 올림
- 증상: `pip install -e source\isaaclab` 후 isaacsim-core(packaging==23.0)·isaaclab_rl(packaging<24) 충돌 경고
- 해결: `pip install --no-deps packaging==23.0`

### 2026-11-19 Isaac Sim 화면(GUI)이 켜진 뒤 꺼짐 (미해결)
- 증상: `isaacsim` 실행 → 창과 UI 가 보였다가 꺼짐. 로그(`kit_20261119_144425.log`, 10,715줄)에 [Error] 0개, 종료 메시지 없이 약 43초 지점에서 끝남. 마지막 기록은 RTX 렌더러의 재질(MDL) 로딩·뷰포트 생성
- 원인: 확인 필요. Isaac Sim 이 설치된 그래픽 드라이버를 검증하지 않은 버전이라고 표시. 포럼에 최신 세대 GPU 에서 Isaac Sim 5.1 GUI 꺼짐을 드라이버를 낮춰 해결한 사례가 있음
- 현재 대응: 학습은 headless 로 정상 동작 (위 시험). 드라이버 변경은 보류. 학습 결과 영상(play)은 D3D12 렌더러로 해결 (아래)

### 2026-11-19 play(영상 녹화)도 렌더링 시작에서 충돌 → D3D12 렌더러로 해결
- 명령: `isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16 --headless --video --video_length 500`
- 증상: 실행 약 9초 만에 `Windows fatal exception: access violation`. 크래시 보고서(dump)가 `isaacsim\kit\data\Kit\Isaac-Sim\5.1\` 에 생성됨
- 위치: Isaac Sim 앱을 만드는 도중, 뷰포트가 RTX 렌더러(Hydra 엔진)를 켜는 단계 (`omni.kit.widget.viewport ... texture.py` 의 `__enable_hydra_engine`). 렌더링용 설정 파일 `isaaclab.python.headless.rendering.kit`, 그래픽 API Vulkan
- 비교: 렌더링 없는 학습(`isaaclab.python.headless.kit`)은 정상이고 그래픽 API 가 D3D12 로 표시됨. GUI 실행(Vulkan)도 꺼졌음 → **RTX 렌더러를 켤 때만** 문제가 생기는 것으로 보임
- 크래시 위치(backtrace): `rtx.scenedb.plugin.dll`
- 원인: Vulkan + RTX 렌더러 + 최신 세대 GPU·드라이버 조합에서 생기는 것으로 보임. 정확한 원인은 확인 필요. 참고: Isaac Sim 5.1 이 테스트한 Windows 드라이버는 580.88 (공식 요구사항 문서). 포럼에 비슷한 GPU 에서 드라이버를 낮춰 해결한 사례와 D3D12 렌더러 강제를 권하는 답변이 있음
- 1차 시도 실패: `isaaclab.bat ... --kit_args="--/renderer/active=d3d12"` → 여전히 `Graphics API: Vulkan` 으로 같은 크래시. 크래시 보고서의 commandLine 에 `--/renderer/active d3d12` 로 `=` 가 빠져 전달됨 (cmd 배치 파일이 인자를 다시 넘기며 `=` 를 구분자로 처리한 것으로 보임, 확인 필요)
- 해결: 환경을 켠 상태에서 `python scripts\...\play.py ... --kit_args "--/app/vulkan=false --/renderer/active=d3d12"` 로 직접 실행 → 시작 표에 `Graphics API: D3D12`, 크래시 없이 영상 저장. 드라이버는 그대로 (다운그레이드 불필요)

### 2026-11-19 동작에서 뺀 관절의 모터 목표값이 0 으로 남음
- 증상: 정책 동작을 다리+허리 13개로 줄이자, 팔 관절이 기본값(어깨 pitch -0.5, 팔꿈치 1.0)이 아닌 0 근처로 처짐. inspect_g1.py 에서 100스텝(2초) 후 팔 목표값 전부 0.000, 실제값 0.009·0.051
- 원인: 관절 목표값은 동작(action) 항목만 설정함. 동작에 없는 관절은 목표값이 초기값 0 그대로 (init_state.joint_pos 는 시작 위치일 뿐 목표값이 아님)
- 해결: 리셋 이벤트 `hold_joints_at_default`(humanoid_tasks/g1_carry/carry_mdp.py)로 팔·손가락 목표값을 기본 자세로 설정 → 재점검에서 8개 환경 모두 리셋 1~2번 후에도 팔 목표값 = 기본값 확인 (2026-11-19)
- 리셋 순서 (원본 코드로 확인): `scene.reset`(Articulation.reset 은 구동기·외력만 초기화, 목표값은 건드리지 않음) → 리셋 이벤트. 동작 항목(JointPositionAction)은 자기 관절 목표값만 씀 → 리셋 이벤트로 넣은 팔 목표값이 덮어써지지 않음

### 2026-11-19 inspect_g1.py 출력을 파일로 저장하면 끝부분이 잘림
- 증상: `python scripts\inspect_g1.py --headless > 파일` 의 출력이 관절 목록 중간에서 끊김, 종료 코드는 0
- 원인: 파일로 보낼 때 stdout 이 버퍼링되는데 `simulation_app.close()` 가 버퍼를 비우기 전에 프로세스를 끝내는 것으로 보임
- 해결: `python -u` (버퍼 없이 출력)

### 2026-11-19 0 동작 점검에서 팔 실제값이 기본값과 정확히 같게 나옴 (착시)
- 증상: 손 3.6kg 인데 100스텝 후 팔꿈치 실제값 1.000 = 기본값. 팔 구동기는 강성 40·감쇠 10 이라 처짐이 보여야 정상
- 원인: 0 동작(다리도 기본 자세 목표)으로는 약 1초(50스텝) 만에 넘어져 리셋됨. 마지막 스텝 직전에 리셋된 환경은 관절 위치가 기본값으로 다시 쓰여 정확히 같게 찍힘 (환경 0: 리셋 2번, 현재 에피소드 0스텝)
- 해결: inspect_g1.py 가 리셋 횟수·현재 에피소드 길이·진행 중 최대 벗어남을 출력하고, 팔 표는 에피소드가 가장 긴 환경 기준으로 출력. 넘어지기 전 값을 보려면 `--steps 30`

### 2026-11-19 0 동작으로는 팔 처짐·출렁임을 제대로 잴 수 없음
- 증상: 리셋 없는 환경만 봐도, 강성 200 인데 40스텝째 어깨 pitch 처짐이 -0.02~+0.03 (계산상 정지 처짐 약 0.1 보다 작음). 강성 40 도 0.35 까지 갔다가 0.12 로 내려감
- 원인: 0 동작이면 착지 후 몸 전체가 기울며 넘어지는 중이라 팔에 걸리는 중력 방향·크기가 계속 바뀜. 팔 PD 응답과 몸 움직임이 섞임
- 해결: `inspect_g1.py --fix_base` (골반을 공중에 고정, `fix_root_link=True`) → 정지 상태에서 중력이 걸리는 계단 응답이 되어 처짐(최종값)과 출렁임(최대/최종)을 바로 읽을 수 있음. 손 질량이 같아야 비교되므로 `--seed`(기본 42) 고정

## 2단계: 짐 든 채 걷기 (Humanoid-G1-Carry-Flat-v0, 2026-11-19 본 학습 완료)
- 목표: 공사장 작업(짐 옮기기·용접)의 첫 단계. 팔은 짐 든 자세로 고정하고 다리 정책이 손 무게가 달라도 걷게 학습
- 구조 결정 (2026-11-19): 하체(걷기·균형)는 강화학습, 상체(잡기·용접)는 이후 IK 제어로 나눔. 학습·시연 모두 Isaac 에서 진행 (다른 시뮬레이터로 정책을 옮기면 물리 엔진·관절 구동 방식 차이로 동작이 보장되지 않음)
- 1차 정책을 그대로 못 쓰는 이유 (원본 코드로 확인): 동작이 `joint_names=[".*"]` 로 팔·손가락까지 37개를 움직이고, 팔에는 `joint_deviation_arms`(-0.1) 약한 벌점만 있음. G1 설정에서 몸통 무게 무작위화(`add_base_mass`)는 꺼져 있음
- G1 Flat 대비 변경 (env_cfg.py)
  - 동작 관절: 다리 12 + 허리 1 (팔·손가락 제외) → 동작 13, 관측 99 (inspect_g1.py 로 확인). 1차 체크포인트 이어받기 불가
  - 팔 시작 자세 `CARRY_POSE` (어깨 pitch -0.5, roll ±0.2, 팔꿈치 pitch 1.0) — 부호 확인함 (어깨 pitch 음수 = 팔이 앞으로, 아래 점검 결과). 원래 G1 기본 자세는 어깨 pitch +0.35, 팔꿈치 0.87
  - 팔·손가락 24개 관절은 `hold_arms` 이벤트(리셋마다)로 모터 목표값을 기본 자세에 고정 (아래 시행착오)
  - 원본 구동기 `arms`(어깨·팔꿈치·손가락 24개 한 그룹, 강성 40·감쇠 10)를 `__post_init__` 에서 둘로 나눔: `arms` 어깨·팔꿈치 10개 강성 200·감쇠 30 (armature 0.01), `fingers` 손가락 14개 강성 40·감쇠 10 (armature 0.001, 원본 값). 새 dict 를 만들어 원본 G1 설정 객체는 건드리지 않음. 이유는 아래 "팔 PD 이득 결정"
  - 손 링크 `left_palm_link`, `right_palm_link` (각 0.281kg, 확인함) 에 0~5kg 추가
  - 몸통(torso_link) 무게중심 ±5cm 무작위화
  - `joint_deviation_arms`, `joint_deviation_fingers` 벌점 제거
  - PPO 설정은 G1 Flat 과 같음 (1500회), 기록 폴더 이름만 `g1_carry_flat`
- G1(g1_minimal.usd) 구성 (inspect_g1.py 출력, 2026-11-19): 링크 44개, 관절 37개. 몸통 torso_link 6.34kg, 골반 pelvis 2.86kg

### 점검 결과 (inspect_g1.py, 2026-11-19)
- 실행: `python -u scripts\inspect_g1.py --headless [--num_envs 8] [--steps 30]`, 원래 G1 과 비교할 때 `--task Isaac-Velocity-Flat-G1-v0`, 팔 PD 비교는 `--fix_base --num_envs 16 --steps 50 [--arm_stiffness K --arm_damping D]`
- 동작 13 (다리 12 + 허리), 관측 99, 리셋 이벤트에 `hold_arms`, 시작 이벤트에 `base_com`·`add_hand_mass` 확인
- 팔 목표값: 8개 환경 모두, 리셋 1~2번 후에도 기본값과 같음 → 통과
- 손 질량: 5회 실행 44개 값이 0.392~5.231kg (기대 범위 0.281~5.281) → 통과
- 팔 실제값 (팔 강성 40 이던 때): 손이 가벼우면 기본값 근처, 무거우면 중력에 눌려 처짐 (PD 제어라 목표값은 맞아도 실제값이 밀림) → 강성을 200 으로 올림 (아래)
  | 조건 | 손 질량 | 어깨 pitch (기본 -0.5) | 팔꿈치 pitch (기본 1.0) |
  |---|---|---|---|
  | 30스텝, 넘어지기 전 | 4.41kg (왼손) | -0.204 | 1.207 |
  | 30스텝, 넘어지기 전 | 0.40kg (오른손) | -0.425 | 1.037 |
  | 100스텝, 에피소드 47스텝째 | 0.88kg | -0.470 | 1.019 |
- 팔 자세 방향 (리셋 직후, 골반 기준 m, x 앞·y 왼쪽·z 위, 왼팔. 오른팔은 y 부호만 반대)
  | 자세 | 어깨 | 팔꿈치 | 손 |
  |---|---|---|---|
  | CARRY_POSE | x -0.002, z +0.259 | x +0.085, z +0.111 | x +0.278, y +0.208, z +0.006 |
  | 원래 G1 | x -0.002, z +0.259 | x -0.070, z +0.101 | x +0.006, y +0.226, z -0.102 |
  - → 짐 든 자세에서 손이 골반 앞 0.28m, 골반 높이, 좌우 간격 0.42m. G1 의 기체 x 축이 앞쪽인 것은 팔꿈치가 굽는 방향(원래 자세에서 손이 팔꿈치보다 앞)과 영상의 무릎·발끝 방향으로 판단

### 팔 PD 이득 결정 (2026-11-19): 어깨·팔꿈치 강성 200, 감쇠 30
- 목표: 손 4~5kg 에서 어깨 pitch 처짐 약 0.1 rad 이내, 출렁임 적게. 손가락은 짐 무게와 관계없으므로 원본(40/10) 유지
- 처짐 계산 (강성): 어깨 pitch 축에서 손까지 앞뒤 거리 0.28m (위 팔 자세 방향 표) → 손 5.28kg(최대) 중력 토크 ≈ 5.28×9.81×0.28 ≈ 14.5Nm, 팔 링크 무게 포함 약 16Nm. 정지 처짐 ≈ 토크/강성 → 강성 40 이면 0.4 rad, 150 이면 0.11, 200 이면 0.08 → 최대 무게에서도 0.1 안팎이 되도록 200
  - 실측(아래)은 0.105 로 계산보다 조금 큼 (팔꿈치도 처지며 지렛대가 길어지는 몫 등으로 보임, 확인 필요)
- 감쇠: 감쇠 비율 ζ = c / (2√(kI)). 실측 넘침 비율로 거꾸로 계산하면 무거운 손(5.25kg) 기준 임계 감쇠가 강성 200 에서 약 40 (등가 관성 약 2 kg·m²) → 30 이면 ζ≈0.7. 가벼운 손은 관성이 작아 더 과감쇠(느리지만 출렁임 없음)
- 실측: `--fix_base --num_envs 16 --steps 50`, 시드 42, 손 5.246kg 인 어깨 pitch (실제-기본, rad)
  | 강성/감쇠 | 50스텝 후 | 최대 | 최대/최종 | 움직임 |
  |---|---|---|---|---|
  | 40/10 (원본) | 0.279 | 0.356 | 1.28 | 처짐 큼, 50스텝에도 덜 멈춤 |
  | 200/10 | 0.091 | 0.132 | 1.42 | 0.132→0.075→0.098 로 출렁임 |
  | 200/20 | 0.098 | 0.115 | 1.17 | 한 번 넘쳤다 돌아옴 |
  | **200/30** | **0.105** | **0.109** | **1.04** | 거의 넘치지 않고 34스텝(0.7초) 이후 일정 |
- 200/30 의 손 질량별 정지 처짐 (어깨 pitch / 팔꿈치 pitch): 4.46kg 0.092/0.053, 4.88kg 0.099/0.057, 5.11~5.25kg 0.101~0.105/0.059, 2.5kg 0.061/0.033, 0.35kg 0.024/0.010
- 서 있는 상태(골반 고정 없음, 30스텝)에서는 착지 충격까지 포함한 최대가 손 4~5kg 에서 0.10~0.12

### 시험 학습 (2026-11-19, 팔 강성 40 이던 때)
- `python -u scripts\train.py --task Humanoid-G1-Carry-Flat-v0 --headless --num_envs 64 --max_iterations 30` → 약 39초, 오류 없음, 총 46,080 스텝
- 평균 보상 -0.46 → -5.38, 에피소드 길이 14 → 36, 마지막 base_contact 1.0 (30회는 동작 확인용이라 모두 넘어짐)
- 기록: `logs\rsl_rl\g1_carry_flat\2026-11-19_17-18-01` (체크포인트 `model_0.pt`, `model_29.pt`)
- play 녹화: `python -u scripts\play.py --task=Humanoid-G1-Carry-Flat-Play-v0 --num_envs 16 --headless --video --video_length 200 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"` → `Graphics API: D3D12`, 크래시 없음, `model_29.pt` 자동 로드, 영상 `videos\play\rl-video-step-0.mp4`
- 영상 프레임: 16대 모두 팔을 앞으로 내밀고 손이 골반 높이인 자세. 옆모습 로봇에서 팔 방향이 무릎·발끝 방향과 같음 → 부호 맞음

### 본 학습 (2026-11-19, 팔 강성 200·감쇠 30)
- `python -u scripts\train.py --task Humanoid-G1-Carry-Flat-v0 --headless --num_envs 4096 --max_iterations 1500` → 32분 21초(학습 루프 31분 41초), 오류 없음, 총 1억 4746만 스텝
- 속도 약 6.5만~8.8만 스텝/초(평균 7.8만, 처음 5회 제외), 반복 1회 평균 1.27초 — 1차(동작 37)와 거의 같음
- 기록: `logs\rsl_rl\g1_carry_flat\2026-11-19_17-33-52` (체크포인트 `model_1499.pt`)
- 지표 변화 (rsl_rl 출력)
  | 지표 | 90회 | 200회 | 600회 | 1000회 | 1499회(마지막) | 1차 G1 Flat 마지막 |
  |---|---|---|---|---|---|---|
  | 평균 보상 | -5.64 | 21.74 | 31.38 | 31.44 | 32.04 | 28.05 |
  | 평균 에피소드 길이 (최대 1000) | 369 | 986 | 1000 | 988 | 984 | 1000 |
  | 이동 속도 따라가기 보상 | 0.274 | 0.905 | 0.926 | 0.935 | 0.934 | 0.925 |
  | 회전 속도 따라가기 보상 | 0.054 | 0.362 | 0.728 | 0.765 | 0.775 | 0.778 |
  | 이동 속도 오차 | 0.52 | 0.29 | 0.24 | 0.22 | 0.23 | 0.245 |
  | 회전 속도 오차 | 3.91 | 1.52 | 0.56 | 0.50 | 0.48 | 0.46 |
  | 넘어짐(base_contact) | 1.0 | 0.0051 | 0.0014 | 0.0031 | 0.0067 | 0.0042 |
  | 동작 노이즈 std | 0.60 | 0.33 | 0.22 | 0.20 | 0.20 | 0.60 |
- 해석: 손에 0~5kg 을 들고도 1차와 비슷한 수준으로 명령 속도를 따라 걸음 (넘어짐 0.1~0.7%). 200회 무렵 이미 넘어지지 않게 되고 이후는 회전 따라가기를 다듬는 단계
  - 평균 보상은 1차와 바로 비교하기 어려움 (팔·손가락 벌점 항목을 뺐으므로). 동작 노이즈 std 도 동작 크기(13 vs 37)가 달라 비교 대상 아님
  - 위 지표는 전체 환경 평균 → 손 무게별로 넘어짐·속도 오차가 다른지는 확인 필요
- play 녹화 (D3D12, 16대, 500스텝 = 10초): `python -u scripts\play.py --task=Humanoid-G1-Carry-Flat-Play-v0 --num_envs 16 --headless --video --video_length 500 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"` → `model_1499.pt` 자동 로드, 크래시 없음, 영상 `videos\play\rl-video-step-0.mp4`
  - 영상 프레임 (2·5·7.6·9.4초): 16대 모두 서서 걸음. 한 로봇을 0.12초 간격 6장(0.6초)으로 보면 다리는 번갈아 내딛고 팔은 앞으로 든 자세 유지, 눈에 띄는 출렁임 없음 (영상 해상도 기준. 걷는 중 팔 관절 벗어남을 숫자로 재지는 않음)

## 미해결 과제
- 2단계 정책 평가: 손 무게별 넘어짐·속도 오차, 걷는 중 팔 관절 벗어남(출렁임)을 숫자로 확인 (현재는 전체 평균 지표와 영상만 확인)
- Isaac Sim GUI 창 실행에도 D3D12 렌더러 지정이 통하는지 확인 필요 (play 영상은 D3D12 로 해결됨)
- 남은 의존성 충돌 경고의 실행 영향 확인 필요
