"""Windows 에서 DLL 을 불러오지 못할 때 뜨는 경고창을 막는다.

LeRobot 이 영상 디코더(TorchCodec)를 불러 보다가 실패하면 "프로시저 시작 지점을 찾을 수 없습니다" 창이 뜬다 (CLAUDE.md 8단계 준비 시행착오).
LeRobot 은 다른 디코더(pyav)로 넘어가므로 실행에는 문제가 없지만, 창은 누군가 닫을 때까지 남는다.
프로세스의 오류 모드를 바꾸면 창을 띄우지 않고 바로 실패로 처리한다. 이 설정은 이 프로세스가 띄우는 하위 프로세스(데이터 읽기 작업자)에도 이어진다.
"""

import sys


def no_error_dialogs() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SEM_NOOPENFILEERRORBOX = 0x0001, 0x0002, 0x8000
    ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)
