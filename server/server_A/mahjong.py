import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

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

PHASE_DISCARD = 0
PHASE_CLAIM = 1
TURN_STABLE_FRAMES = 5
TURN_CORRECTION_COOLDOWN_FRAMES = 15
OBS_DIM = 175
ACTION_DIM = 39

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
    "EAST": 27,
    "SOUTH": 28,
    "WEST": 29,
    "NORTH": 30,
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


def tile_labels_to_ids(labels: Sequence[str]) -> List[int]:
    """Convert classifier labels to legal 0..33 tile ids."""
    tile_ids: List[int] = []
    for label in labels:
        tile_id = tile_label_to_id(label)
        if tile_id is not None and 0 <= tile_id < 34:
            tile_ids.append(int(tile_id))
    return tile_ids


def sort_tile_ids(tiles: Sequence[int]) -> List[int]:
    """Sort tile ids by suit and rank using the fixed 0..33 action/order mapping."""
    valid_tiles: List[int] = []
    for tile in tiles:
        tile_id = int(tile)
        if 0 <= tile_id < 34:
            valid_tiles.append(tile_id)
    return sorted(valid_tiles)


def sorted_tile_labels_from_ids(tiles: Sequence[int]) -> List[str]:
    """Return display labels sorted as m, p, s, honors."""
    return [tile_id_to_label(tile_id) for tile_id in sort_tile_ids(tiles)]


def sort_tile_labels(labels: Sequence[str]) -> List[str]:
    """Sort classifier labels by Mahjong suit/rank; unknown labels keep their relative order at the end."""
    sortable = []
    unknown = []
    for idx, label in enumerate(labels):
        tile_id = tile_label_to_id(label)
        if tile_id is None:
            unknown.append((idx, str(label)))
        else:
            sortable.append((int(tile_id), idx, str(label)))
    sortable.sort(key=lambda item: (item[0], item[1]))
    return [label for _, _, label in sortable] + [label for _, label in unknown]


def is_self_discard_turn_by_hand_count(hand_tiles: Sequence[int]) -> bool:
    """Player0 has just drawn when hand count is 3n+2."""
    return len(hand_tiles) > 0 and len(hand_tiles) % 3 == 2


def count34_from_tiles(tiles: Sequence[int]) -> np.ndarray:
    """Build a 34-dim tile count vector from 0..33 tile ids."""
    cnt = np.zeros((34,), dtype=np.float32)
    for tile in tiles:
        tile_id = int(tile)
        if 0 <= tile_id < 34:
            cnt[tile_id] += 1.0
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


def fallback_discard_tile(detected_hand_tiles: Sequence[int]) -> Optional[int]:
    """Return a legal discard tile from the detected hand."""
    if not detected_hand_tiles:
        return None

    cnt = count34_from_tiles(detected_hand_tiles)
    tile_id = heuristic_safe_discard_id(cnt)
    if tile_id is not None and int(cnt[int(tile_id)]) > 0:
        return int(tile_id)

    for tile in detected_hand_tiles:
        tile_id = int(tile)
        if 0 <= tile_id < 34:
            return tile_id
    return None


class GameStateTracker:
    """Minimal state needed to build the PPO observation from live vision."""

    def __init__(
        self,
        stable_frames: int = TURN_STABLE_FRAMES,
        cooldown_frames: int = TURN_CORRECTION_COOLDOWN_FRAMES,
    ):
        self.current_player = 0
        self.phase = PHASE_DISCARD
        self.discards: List[List[int]] = [[], [], [], []]
        self.last_discard: Optional[int] = None
        self.last_discarder: Optional[int] = None
        self.open_chi_count = 0
        self.open_pon_count = 0
        self.frame_index = 0
        self.cooldown_frames_remaining = 0
        self._recent_hand_counts = deque(maxlen=max(1, int(stable_frames)))
        self._stable_frames = max(1, int(stable_frames))
        self._cooldown_frames = max(0, int(cooldown_frames))

    def reset_runtime(self) -> None:
        self.frame_index = 0
        self.cooldown_frames_remaining = 0
        self._recent_hand_counts.clear()

    def update_hand_count_stability(self, detected_hand_tiles: Sequence[int]) -> bool:
        self.frame_index += 1
        if self.cooldown_frames_remaining > 0:
            self.cooldown_frames_remaining -= 1

        self._recent_hand_counts.append(len(detected_hand_tiles))
        if len(self._recent_hand_counts) < self._stable_frames:
            return False
        return len(set(self._recent_hand_counts)) == 1

    def can_correct_this_frame(self) -> bool:
        return self.cooldown_frames_remaining <= 0

    def correct_to_self_discard(self) -> None:
        self.current_player = 0
        self.phase = PHASE_DISCARD
        self.cooldown_frames_remaining = self._cooldown_frames


def build_obs_from_tracker(tracker: GameStateTracker, detected_hand_tiles: Sequence[int]) -> np.ndarray:
    """Build the unchanged 175-dim PPO observation layout."""
    obs = np.zeros((OBS_DIM,), dtype=np.float32)

    obs[0:34] = count34_from_tiles(detected_hand_tiles)

    offset = 34
    for player in range(4):
        obs[offset : offset + 34] = count34_from_tiles(tracker.discards[player])
        offset += 34

    obs[170] = 0.0 if tracker.last_discard is None else (float(tracker.last_discard) + 1.0) / 34.0
    obs[171] = 0.0 if tracker.last_discarder is None else (float(tracker.last_discarder) + 1.0) / 4.0
    obs[172] = float(tracker.phase)
    obs[173] = float(tracker.open_chi_count) / 5.0
    obs[174] = float(tracker.open_pon_count) / 5.0

    return obs


def _topk_ppo_action_probs(rl_model, obs: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    if rl_model is None or torch is None or top_k <= 0:
        return []
    try:
        device = getattr(rl_model, "device", "cpu")
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = rl_model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.squeeze(0).detach().cpu().numpy()
        idx = np.argsort(-probs)[:top_k]
        return [
            {
                "action": int(action),
                "prob": float(probs[action]),
                "tile": tile_id_to_label(int(action)) if 0 <= int(action) < 34 else action_to_label(int(action)),
            }
            for action in idx
        ]
    except Exception as e:
        return [{"action": -1, "prob": 0.0, "tile": "", "error": str(e)}]


def action_to_label(action: int) -> str:
    if 0 <= action <= 33:
        return tile_id_to_label(action)
    if action == 34:
        return "PASS"
    if action == 35:
        return "PON"
    if action == 36:
        return "CHI_LOW"
    if action == 37:
        return "CHI_MID"
    if action == 38:
        return "CHI_HIGH"
    return str(action)


def _empty_live_agent_result(detected_hand_tiles: Sequence[int], reason: str) -> Dict[str, Any]:
    return {
        "corrected": False,
        "stable_count_frames": 0,
        "hand_count": len(detected_hand_tiles),
        "recommended_action": -1,
        "recommended_tile": "",
        "source": "none",
        "reason": reason,
        "topk": [],
    }


def maybe_correct_self_turn_and_suggest(
    tracker: GameStateTracker,
    detected_hand_tiles: Sequence[int],
    rl_model=None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Correct Player0 discard phase from stable hand count and produce a legal discard."""
    hand_tiles = [int(t) for t in detected_hand_tiles if 0 <= int(t) < 34]
    stable = tracker.update_hand_count_stability(hand_tiles)

    if not stable:
        result = _empty_live_agent_result(hand_tiles, "hand count is not stable yet")
        result["stable_count_frames"] = len(tracker._recent_hand_counts)
        return result

    if not is_self_discard_turn_by_hand_count(hand_tiles):
        result = _empty_live_agent_result(hand_tiles, "stable hand count is not 3n+2")
        result["stable_count_frames"] = len(tracker._recent_hand_counts)
        return result

    corrected = False
    if tracker.can_correct_this_frame():
        tracker.correct_to_self_discard()
        corrected = True

    obs = build_obs_from_tracker(tracker, hand_tiles)
    source = "ppo"
    reason = "PPO policy (deterministic) after Player0 discard-turn correction"
    action_id: Optional[int] = None

    if rl_model is not None:
        try:
            action, _ = rl_model.predict(obs, deterministic=True)
            action_id = int(action)
        except Exception as e:
            source = "fallback"
            reason = f"PPO predict failed; fallback discard ({e})"
            action_id = fallback_discard_tile(hand_tiles)
    else:
        source = "fallback"
        reason = "PPO model not loaded; fallback discard"
        action_id = fallback_discard_tile(hand_tiles)

    if action_id is None or not (0 <= action_id <= 33) or action_id not in hand_tiles:
        source = "fallback" if source == "ppo" else source
        reason = "PPO action illegal for detected hand; fallback discard"
        action_id = fallback_discard_tile(hand_tiles)

    tile = tile_id_to_label(action_id) if action_id is not None else ""

    return {
        "corrected": corrected,
        "current_player": tracker.current_player,
        "phase": tracker.phase,
        "stable_count_frames": len(tracker._recent_hand_counts),
        "cooldown_frames_remaining": tracker.cooldown_frames_remaining,
        "hand_count": len(hand_tiles),
        "recommended_action": int(action_id) if action_id is not None else -1,
        "recommended_tile": tile,
        "source": source,
        "reason": reason,
        "topk": _topk_ppo_action_probs(rl_model, obs, top_k=top_k),
    }


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
