"""방별 대화 상태 (서버 메모리).

방마다 보관하는 것
  - 최근 대화 턴: 말한 사람, 영어 원문, 한국어 번역, 시각 (최대 ROOM_MAX_TURNS 개)
  - 롤링 요약: 사용자별로 따로 ({"remote": [...], "me": [...]} 는 보는 사람 기준이라서)
  - 대화 도메인: 사전 검색에서 쓰는 직전 도메인

주의: 서버를 재시작하면 전부 사라진다. 통화 전체 기록은 따로 저장(Postgres)할 것.
오래 안 쓴 방(ROOM_IDLE_TTL_SEC)은 자동으로 지운다.
"""
import time
from collections import deque
from dataclasses import dataclass, field

from config import DEFAULT_DOMAIN, ROOM_IDLE_TTL_SEC, ROOM_MAX_TURNS

UNKNOWN = "unknown"   # 앱이 speaker 를 안 보냈을 때


@dataclass
class Turn:
    speaker: str        # 말한 사람 (앱이 보낸 사용자 id)
    text: str           # 영어 원문
    translation: str    # 한국어 번역
    ts: float           # 시각 (epoch 초)


@dataclass
class RoomState:
    turns: deque = field(default_factory=lambda: deque(maxlen=ROOM_MAX_TURNS))
    summaries: dict = field(default_factory=dict)   # 사용자 id → {"remote": [...], "me": [...]}
    domain: str = DEFAULT_DOMAIN
    last_active: float = field(default_factory=time.time)


_ROOMS: dict[str, RoomState] = {}


def _purge_idle(now: float):
    stale = [rid for rid, r in _ROOMS.items() if now - r.last_active > ROOM_IDLE_TTL_SEC]
    for rid in stale:
        del _ROOMS[rid]


def get_room(room_id: str) -> RoomState:
    """방 상태를 돌려준다. 없으면 새로 만든다."""
    now = time.time()
    _purge_idle(now)
    room = _ROOMS.get(room_id)
    if room is None:
        room = _ROOMS[room_id] = RoomState()
    room.last_active = now
    return room


def room_count() -> int:
    return len(_ROOMS)


# ---------- 대화 턴 ----------
def add_turn(room_id: str, speaker: str | None, text: str, translation: str) -> Turn:
    turn = Turn(speaker=speaker or UNKNOWN, text=text, translation=translation, ts=time.time())
    get_room(room_id).turns.append(turn)
    return turn


def recent_turns(room_id: str, n: int) -> list[Turn]:
    """최근 n개 턴 (오래된 것 → 최신 순서)."""
    if n <= 0:
        return []
    return list(get_room(room_id).turns)[-n:]


# ---------- 롤링 요약 (사용자별) ----------
def get_summary(room_id: str, user_id: str | None) -> dict:
    saved = get_room(room_id).summaries.get(user_id or UNKNOWN)
    if not saved:
        return {"remote": [], "me": []}
    return {"remote": list(saved.get("remote", [])), "me": list(saved.get("me", []))}


def set_summary(room_id: str, user_id: str | None, summary: dict):
    get_room(room_id).summaries[user_id or UNKNOWN] = {
        "remote": list(summary.get("remote", [])),
        "me": list(summary.get("me", [])),
    }


# ---------- 대화 도메인 ----------
def get_domain(room_id: str) -> str:
    return get_room(room_id).domain


def set_domain(room_id: str, domain: str):
    if domain:
        get_room(room_id).domain = domain
