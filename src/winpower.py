"""Windows 전원 관리 대응: 무인 장시간 작업(학습·추출)이 느려지거나 절전으로 끊기지 않게 이 프로세스만 예외를 요청한다.
시스템 설정은 바꾸지 않으며 프로세스가 끝나면 풀린다. 다른 OS 에서는 아무것도 안 한다."""
import os


def no_power_throttling():
    """Windows 11 은 사용자가 자리를 비우면 창 없는 백그라운드 프로세스를 효율 모드(EcoQoS: 저전력 코어·낮은 클럭)로 돌린다.
    이 프로세스만 그 대상에서 빼 달라고 요청한다(시스템 설정은 바꾸지 않고 프로세스가 끝나면 사라짐). 다른 OS 에서는 아무것도 안 한다.
    2026-10-19 사용자 부재 중 학습 1회가 평소의 5~7배로 느려진 일 때문에 넣었다(T-023)."""
    if os.name != "nt":
        return False
    import ctypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    s = _State(1, 0x1, 0x0)          # PROCESS_POWER_THROTTLING_EXECUTION_SPEED 를 제어하되 끈 상태로
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p          # 64비트 핸들이 잘리지 않게 타입을 명시
    k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k.SetProcessInformation.restype = ctypes.c_int
    ok = bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))   # 4 = ProcessPowerThrottling
    if not ok:
        print("효율 모드 제외 요청 실패, 오류 코드", ctypes.get_last_error(), flush=True)
    return ok


def keep_awake():
    """학습이 도는 동안 시스템이 절전(모던 스탠바이)으로 들어가지 않게 요청한다(ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
    동영상 재생 프로그램이 쓰는 것과 같은 방식이며 시스템 설정은 바꾸지 않고 프로세스가 끝나면 풀린다. 덮개를 닫는 것은 막지 못한다.
    2026-10-20 00:44 절전 진입 중 비정상 재부팅으로 실험이 끊긴 일 때문에 넣었다(T-026)."""
    if os.name != "nt":
        return False
    import ctypes
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000 | 0x00000001)))
