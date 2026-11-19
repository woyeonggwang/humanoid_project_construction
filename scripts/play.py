"""학습 결과 재생·녹화. 예)
python scripts\\play.py --task=Humanoid-G1-Carry-Flat-Play-v0 --num_envs 16 --headless --video --video_length 500 --kit_args "--/app/vulkan=false --/renderer/active=d3d12"
"""

from _launch import run

run("play.py")
