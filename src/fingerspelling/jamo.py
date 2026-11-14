"""현대 한글 Unicode 자모 처리. 실제 수어 지문자 37종과의 대응은 정의하지 않는다.

호환 자모를 사용하며 겹받침·쌍자음·복합 모음은 쪼개지 않는다.
CTC blank, 반복 제거, 사전 기반 단어 추정은 이 모듈의 책임이 아니다.
"""
import unicodedata

INITIALS = tuple("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
MEDIALS = tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
FINALS = ("",) + tuple("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
JAMO = frozenset(INITIALS + MEDIALS + FINALS[1:])
BOUNDARY = "|"


def decompose_hangul(text: str, *, preserve_boundaries: bool = False) -> list[str]:
    """문자열을 호환 자모 리스트로 분리한다. 빈 문자열은 []를 반환한다.

    NFC 정규화 후 현대 완성형 음절과 독립 호환 자모만 허용한다.
    따라서 NFD 음절도 지원하나 고립된 결합 자모·옛한글·공백·숫자·영문은
    ValueError, 문자열 이외 입력은 TypeError다. 자동 삭제·치환은 하지 않는다.
    preserve_boundaries=True이면 정규화된 원문 문자 사이에 '|'를 넣는다.
    이 모드는 독립 호환 자모를 포함해 NFC 원문으로 정확히 왕복한다.
    기본 모드는 문자 경계와 자모의 초성/종성 역할 정보를 잃는다.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if not isinstance(preserve_boundaries, bool):
        raise TypeError("preserve_boundaries must be bool")
    result = []
    for index, char in enumerate(unicodedata.normalize("NFC", text)):
        if preserve_boundaries and index:
            result.append(BOUNDARY)
        code = ord(char) - 0xAC00
        if 0 <= code < 11172:
            initial, rest = divmod(code, 588)
            medial, final = divmod(rest, 28)
            result.extend((INITIALS[initial], MEDIALS[medial]))
            if final:
                result.append(FINALS[final])
        elif char in JAMO:
            result.append(char)
        else:
            raise ValueError(f"unsupported character at {index}: {char!r}")
    return result


def _syllable(tokens: list[str]) -> str:
    if (len(tokens) not in (2, 3) or tokens[0] not in INITIALS
            or tokens[1] not in MEDIALS
            or (len(tokens) == 3 and tokens[2] not in FINALS[1:])):
        raise ValueError(f"invalid syllable: {tokens!r}")
    final = FINALS.index(tokens[2]) if len(tokens) == 3 else 0
    return chr(0xAC00 + INITIALS.index(tokens[0]) * 588
                + MEDIALS.index(tokens[1]) * 28 + final)


def compose_hangul(jamo: list[str], *, preserve_boundaries: bool = False) -> str:
    """호환 자모 리스트를 완성형 문자열로 조합한다. []는 ''를 반환한다.

    기본 모드는 (초성, 중성, 선택적 종성)의 완전한 음절열만 허용한다.
    중성 뒤 자음이 바로 다음 중성의 초성이면 다음 음절에 배정한다.
    겹받침·복합 모음을 합치거나 빠진 ㅇ을 넣지 않는다. 독립 자모를
    허용하지 않으므로 이 결과는 사전/언어 모델로 검증된 단어가 아니다.
    예: ㄱㅏㄴㅏ → 가나; 원문 '간ㅏ'를 의도했다면 경계 모드가 필요하다.

    preserve_boundaries=True이면 '|'로 나눈 각 그룹은 한 음절 또는
    독립 호환 자모 하나여야 한다. 빈 그룹은 오류다. 경계 없는 입력도
    이 모드에서는 한 그룹이다. decompose의 같은 옵션과 짝지어 사용한다.
    잘못된 컨테이너/원소 타입은 TypeError, 미지원 토큰·잘못된 배열은
    ValueError다. 결합 자모는 받지 않으며 입력 리스트를 수정하지 않는다.
    """
    if not isinstance(jamo, list):
        raise TypeError("jamo must be list[str]")
    if not isinstance(preserve_boundaries, bool):
        raise TypeError("preserve_boundaries must be bool")
    for index, token in enumerate(jamo):
        if not isinstance(token, str):
            raise TypeError(f"token at {index} must be str")
        if token not in JAMO and not (preserve_boundaries and token == BOUNDARY):
            raise ValueError(f"unsupported token at {index}: {token!r}")
    if not jamo:
        return ""
    output = []
    if preserve_boundaries:
        start = 0
        for end in range(len(jamo) + 1):
            if end == len(jamo) or jamo[end] == BOUNDARY:
                group = jamo[start:end]
                if not group:
                    raise ValueError("empty boundary group")
                output.append(group[0] if len(group) == 1 else _syllable(group))
                start = end + 1
    else:
        start = 0
        while start < len(jamo):
            end = start + 2
            if end < len(jamo) and jamo[end] in FINALS[1:]:
                next_initial = (jamo[end] in INITIALS and end + 1 < len(jamo)
                                and jamo[end + 1] in MEDIALS)
                if not next_initial:
                    end += 1
            output.append(_syllable(jamo[start:end]))
            start = end
    return "".join(output)
