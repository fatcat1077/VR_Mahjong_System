import re
from typing import Any, Dict, List, Optional

import numpy as np

# Optional: SB3 PPO Mahjong helper
try:
    from stable_baselines3 import PPO  # type: ignore
except Exception:
    PPO = None  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore


# ============================================================
# Mahjong tile label <-> 34-id utilities (for PPO model)
# 34 IDs (common Riichi / Tenhou order):
#   0-8:  1m-9m
#   9-17: 1p-9p
#   18-26:1s-9s
#   27-33: East,South,West,North,White,Green,Red
# ============================================================

_CANONICAL_ID_TO_LABEL = (
    [f"{i}m" for i in range(1, 10)]
    + [f"{i}p" for i in range(1, 10)]
    + [f"{i}s" for i in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
)

_ZH_NUM = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_HONOR_MAP = {
    "E": 27,
    "S": 28,
    "W": 29,
    "N": 30,
    "P": 31,
    "F": 32,
    "C": 33,
    # common alternates
    "WHITE": 31,
    "WH": 31,
    "WD": 31,
    "BAI": 31,
    "GREEN": 32,
    "GD": 32,
    "FA": 32,
    "RED": 33,
    "RD": 33,
    "ZHONG": 33,
    "R": 33,
    # Chinese
    "東": 27,
    "南": 28,
    "西": 29,
    "北": 30,
    "白": 31,
    "發": 32,
    "发": 32,
    "中": 33,
}

_SUIT_ALIASES = {
    "m": ("m", "man", "wan", "萬", "万"),
    "p": ("p", "pin", "tong", "筒", "餅", "饼", "pinzu", "pzu"),
    "s": ("s", "sou", "suo", "索", "条", "條", "souzu", "szu"),
}


def tile_label_to_id(label: str) -> Optional[int]:
    """Convert classifier label to tile id (0..33).

    Supports common formats:
      - '1m','9p','3s'
      - '1萬','九萬','3筒','7索','7條'
      - honors: 'E/S/W/N/P/F/C', or '東南西北白發中'
      - numeric string '0'..'33'
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None

    # numeric id
    if s.isdigit():
        v = int(s)
        return v if 0 <= v < 34 else None

    # direct honors
    up = s.upper().strip()
    if up in _HONOR_MAP:
        return _HONOR_MAP[up]
    if s in _HONOR_MAP:
        return _HONOR_MAP[s]

    s2 = s.replace(" ", "").replace("_", "")
    up2 = s2.upper()
    if up2 in _HONOR_MAP:
        return _HONOR_MAP[up2]
    if s2 in _HONOR_MAP:
        return _HONOR_MAP[s2]

    # suit like 1m / 2p / 3s
    m = re.match(r"^([1-9])([mpsMPS])$", s2)
    if m:
        num = int(m.group(1))
        suit = m.group(2).lower()
        base = {"m": 0, "p": 9, "s": 18}[suit]
        return base + (num - 1)

    # chinese number + suit: 一萬 / 3萬 / 九筒 / 7索 / 7條
    num = None
    if len(s2) >= 2 and s2[0] in _ZH_NUM:
        num = _ZH_NUM[s2[0]]
        rest = s2[1:]
    else:
        m2 = re.match(r"^([1-9])(.+)$", s2)
        if m2:
            num = int(m2.group(1))
            rest = m2.group(2)
        else:
            return None

    if num is None or not (1 <= num <= 9):
        return None

    for suit, aliases in _SUIT_ALIASES.items():
        for a in aliases:
            if rest == a or rest.endswith(a):
                base = {"m": 0, "p": 9, "s": 18}[suit]
                return base + (num - 1)

    return None


def tile_id_to_label(tile_id: int) -> str:
    if 0 <= int(tile_id) < 34:
        return _CANONICAL_ID_TO_LABEL[int(tile_id)]
    return str(tile_id)


def choose_display_label(tile_id: int, original_labels: List[str]) -> str:
    """Prefer a label already in the hand; otherwise fallback to canonical."""
    for lb in original_labels:
        tid = tile_label_to_id(lb)
        if tid == tile_id:
            return lb
    return tile_id_to_label(tile_id)


def obs34_from_hand_labels(hand_labels: List[str]) -> np.ndarray:
    cnt = np.zeros((34,), dtype=np.int8)
    for lb in hand_labels:
        tid = tile_label_to_id(lb)
        if tid is None:
            continue
        cnt[tid] += 1
    return cnt


def heuristic_safe_discard_id(cnt: np.ndarray) -> Optional[int]:
    """"Most safe" heuristic without table context.

    1) honors (27..33) singletons, then honors overall
    2) isolated terminals (1/9), then terminals overall
    3) isolated simples
    """
    if cnt is None or cnt.shape[0] != 34:
        return None

    def in_same_suit(a: int, b: int) -> bool:
        if a >= 27 or b >= 27:
            return False
        return (a // 9) == (b // 9)

    def connectedness(t: int) -> int:
        # higher = more useful to keep
        c = int(cnt[t])
        if t >= 27:
            return c * 3
        for d in (-2, -1, 1, 2):
            u = t + d
            if 0 <= u < 27 and in_same_suit(t, u):
                c += int(cnt[u])
        return c

    def is_terminal(t: int) -> bool:
        if t >= 27:
            return False
        n = (t % 9) + 1
        return n == 1 or n == 9

    candidates = [i for i in range(34) if cnt[i] > 0]
    if not candidates:
        return None

    def safety_rank(t: int) -> int:
        if t >= 27:
            return 0
        if is_terminal(t):
            return 1
        return 2

    def singleton_bonus(t: int) -> int:
        return 0 if cnt[t] == 1 else 1

    candidates.sort(key=lambda t: (safety_rank(t), singleton_bonus(t), connectedness(t), -int(cnt[t]), t))
    return candidates[0]


def heuristic_benefit_fallback_id(cnt: np.ndarray) -> Optional[int]:
    """When PPO is unavailable/illegal action.

    Currently identical to safe heuristic.
    """
    return heuristic_safe_discard_id(cnt)


def load_ppo_model(ppo_model_path: Optional[str], ppo_device: Optional[str] = None):
    """Load SB3 PPO model if available. Returns model or None."""
    if not ppo_model_path:
        return None
    if PPO is None:
        print("[PPO] stable-baselines3 not installed. Install with: pip install stable-baselines3")
        return None

    try:
        if ppo_device is not None:
            dev = ppo_device
        else:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        model = PPO.load(ppo_model_path, device=dev)
        print(f"[PPO] Loaded PPO model: {ppo_model_path} (device={dev})")
        return model
    except Exception as e:
        print(f"[PPO] Failed to load model '{ppo_model_path}': {e}")
        return None


def make_advice(hand_labels: List[str], ppo_model=None) -> Dict[str, Any]:
    """Return advice dict (benefit via PPO, safe via heuristic)."""
    cnt = obs34_from_hand_labels(hand_labels)

    benefit_id: Optional[int] = None
    benefit_src = "ppo"
    benefit_reason = "PPO policy (deterministic)"

    if ppo_model is not None:
        try:
            obs = cnt.astype(np.float32)
            action, _ = ppo_model.predict(obs, deterministic=True)
            benefit_id = int(action)
            if benefit_id < 0 or benefit_id >= 34 or cnt[benefit_id] <= 0:
                benefit_reason = "PPO action illegal for current hand; fallback heuristic"
                benefit_src = "fallback"
                benefit_id = heuristic_benefit_fallback_id(cnt)
        except Exception as e:
            benefit_reason = f"PPO predict failed; fallback heuristic ({e})"
            benefit_src = "fallback"
            benefit_id = heuristic_benefit_fallback_id(cnt)
    else:
        benefit_src = "fallback"
        benefit_reason = "PPO model not loaded; fallback heuristic"
        benefit_id = heuristic_benefit_fallback_id(cnt)

    safe_id = heuristic_safe_discard_id(cnt)
    safe_src = "heuristic"
    safe_reason = "No table context; heuristic prefers honors/terminals & least-connected tile"

    benefit_tile = choose_display_label(benefit_id, hand_labels) if benefit_id is not None else ""
    safe_tile = choose_display_label(safe_id, hand_labels) if safe_id is not None else ""

    return {
        "benefit": {
            "tile_id": int(benefit_id) if benefit_id is not None else -1,
            "tile": benefit_tile,
            "source": benefit_src,
            "reason": benefit_reason,
        },
        "safe": {
            "tile_id": int(safe_id) if safe_id is not None else -1,
            "tile": safe_tile,
            "source": safe_src,
            "reason": safe_reason,
        },
    }
