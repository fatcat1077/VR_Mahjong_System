import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional: SB3 PPO Mahjong helper
try:
    from stable_baselines3 import PPO  # type: ignore
except Exception:
    PPO = None  # type: ignore

try:
    from sb3_contrib import MaskablePPO  # type: ignore
except Exception:
    MaskablePPO = None  # type: ignore

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
    + ["east", "south", "west", "north", "white", "green", "red"]
)

PHASE_DISCARD = 0
PHASE_CLAIM = 1
TURN_STABLE_FRAMES = 7
TABLE_STABLE_FRAMES = 4
TURN_CORRECTION_COOLDOWN_FRAMES = 15
OBS_DIM = 175
ACTION_DIM = 39
ACT_PASS = 34
ACT_PON = 35
ACT_CHI_LOW = 36
ACT_CHI_MID = 37
ACT_CHI_HIGH = 38

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
    """Live game-state debouncer used to build PPO observations from vision."""

    def __init__(
        self,
        stable_frames: int = TURN_STABLE_FRAMES,
        cooldown_frames: int = TURN_CORRECTION_COOLDOWN_FRAMES,
        table_pos_bucket: float = 0.025,
        table_match_dist: float = 0.12,
    ):
        self._stable_frames = max(1, int(stable_frames))
        self._table_stable_frames = max(1, min(self._stable_frames, TABLE_STABLE_FRAMES))
        self._cooldown_frames = max(0, int(cooldown_frames))
        self._table_pos_bucket = max(0.005, float(table_pos_bucket))
        self._table_match_dist = max(0.01, float(table_match_dist))
        self._recent_hand_counts = deque(maxlen=self._stable_frames)
        self.reset_runtime()

    def reset_runtime(self) -> None:
        self.current_player = 0
        self.phase = PHASE_DISCARD
        self.discards: List[List[int]] = [[], [], [], []]
        self.last_discard: Optional[int] = None
        self.last_discarder: Optional[int] = None
        self.open_chi_count = 0
        self.open_pon_count = 0
        self.frame_index = 0
        self.cooldown_frames_remaining = 0
        self._expected_discarder = 0
        self._self_discard_turn_active = False
        self._stable_hand_count: Optional[int] = None
        self._recent_hand_counts.clear()
        self._table_candidate_signature: Optional[Tuple[Tuple[int, int], ...]] = None
        self._table_candidate_frames = 0
        self._table_candidate_obs: List[Dict[str, Any]] = []
        self._stable_table_signature: Optional[Tuple[Tuple[int, int], ...]] = None
        self._stable_table_obs: List[Dict[str, Any]] = []
        self._reset_table_baseline_on_next_stable = False
        self._pending_self_table_tile: Optional[int] = None
        self._last_live_info: Dict[str, Any] = {}

    def _normalize_table_observations(self, table_observations: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for obs in table_observations or []:
            tile_id = obs.get("tile_id")
            if tile_id is None:
                tile_id = tile_label_to_id(str(obs.get("label", "")))
            if tile_id is None or not (0 <= int(tile_id) < 34):
                continue
            normalized.append(
                {
                    "track_id": int(obs.get("track_id", obs.get("id", -1))),
                    "tile_id": int(tile_id),
                    "label": str(obs.get("label") or tile_id_to_label(int(tile_id))),
                    "cx": float(obs.get("cx", 0.0)),
                    "cy": float(obs.get("cy", 0.0)),
                    "conf": float(obs.get("conf", 0.0)),
                }
            )
        normalized.sort(key=lambda item: (item["cx"], item["cy"], item["tile_id"], item["track_id"]))
        return normalized

    def _table_signature(self, table_observations: Sequence[Dict[str, Any]]) -> Tuple[Tuple[int, int], ...]:
        counts: Dict[int, int] = {}
        for obs in table_observations:
            tile_id = int(obs["tile_id"])
            counts[tile_id] = counts.get(tile_id, 0) + 1
        return tuple(sorted(counts.items()))

    @staticmethod
    def _obs_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        dx = float(a.get("cx", 0.0)) - float(b.get("cx", 0.0))
        dy = float(a.get("cy", 0.0)) - float(b.get("cy", 0.0))
        return float((dx * dx + dy * dy) ** 0.5)

    def _find_new_table_observations(
        self,
        previous: Sequence[Dict[str, Any]],
        current: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        unmatched = list(range(len(current)))

        # First keep identities when the tracker ID survives.
        for prev in previous:
            prev_tid = int(prev.get("track_id", -1))
            if prev_tid < 0:
                continue
            matched_idx = None
            for idx in unmatched:
                cur = current[idx]
                if int(cur.get("track_id", -2)) == prev_tid and self._obs_distance(prev, cur) <= self._table_match_dist:
                    matched_idx = idx
                    break
            if matched_idx is not None:
                unmatched.remove(matched_idx)

        # Then match same-label nearby tiles. This handles tracker-ID churn.
        for prev in previous:
            best_idx = None
            best_dist = self._table_match_dist
            for idx in unmatched:
                cur = current[idx]
                if int(cur["tile_id"]) != int(prev["tile_id"]):
                    continue
                dist = self._obs_distance(prev, cur)
                if dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            if best_idx is not None:
                unmatched.remove(best_idx)

        new_items = [current[idx] for idx in unmatched]
        new_items.sort(key=lambda item: (item["cx"], item["cy"], item["tile_id"]))
        return new_items

    def can_correct_this_frame(self) -> bool:
        return self.cooldown_frames_remaining <= 0

    def correct_to_self_discard(self) -> None:
        self.current_player = 0
        self.phase = PHASE_DISCARD
        self._expected_discarder = 0
        self._self_discard_turn_active = True
        self._reset_table_baseline_on_next_stable = True
        self.cooldown_frames_remaining = self._cooldown_frames

    def _accept_new_discard(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        tile_id = int(obs["tile_id"])
        discarder = int(self._expected_discarder)
        self.discards[discarder].append(tile_id)
        self.last_discard = tile_id
        self.last_discarder = discarder
        self._expected_discarder = (discarder + 1) % 4
        self.current_player = self._expected_discarder

        if discarder == 0:
            self.phase = PHASE_DISCARD
            self._self_discard_turn_active = False
        else:
            self.phase = PHASE_CLAIM

        return {
            "type": "table_discard",
            "discarder": discarder,
            "tile_id": tile_id,
            "tile": tile_id_to_label(tile_id),
            "track_id": int(obs.get("track_id", -1)),
            "cx": float(obs.get("cx", 0.0)),
            "cy": float(obs.get("cy", 0.0)),
        }

    def complete_self_discard_from_hand(self, tile_id: int) -> Dict[str, Any]:
        tile_id = int(tile_id)
        self.discards[0].append(tile_id)
        self.last_discard = tile_id
        self.last_discarder = 0
        self._expected_discarder = 1
        self.current_player = 1
        self.phase = PHASE_DISCARD
        self._self_discard_turn_active = False
        self._pending_self_table_tile = tile_id
        self._reset_table_baseline_on_next_stable = False
        return {
            "type": "self_discard_left_hand",
            "discarder": 0,
            "tile_id": tile_id,
            "tile": tile_id_to_label(tile_id),
        }

    def _accept_pending_self_table_baseline(self, cur_obs: List[Dict[str, Any]], new_obs: List[Dict[str, Any]]) -> Dict[str, Any]:
        expected_tile = self._pending_self_table_tile
        matched = any(int(obs.get("tile_id", -1)) == int(expected_tile) for obs in new_obs) if expected_tile is not None else False
        self._stable_table_signature = self._table_signature(cur_obs)
        self._stable_table_obs = cur_obs
        self._pending_self_table_tile = None
        return {
            "type": "table_self_discard_baseline",
            "tile_id": int(expected_tile) if expected_tile is not None else -1,
            "tile": tile_id_to_label(int(expected_tile)) if expected_tile is not None else "",
            "matched": bool(matched),
            "count": len(cur_obs),
        }

    def stable_table_labels(self) -> List[str]:
        """Return the locked stable table state for display."""
        return sorted_tile_labels_from_ids([int(obs["tile_id"]) for obs in self._stable_table_obs])

    def update_live_state(
        self,
        detected_hand_tiles: Sequence[int],
        table_observations: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.frame_index += 1
        if self.cooldown_frames_remaining > 0:
            self.cooldown_frames_remaining -= 1

        hand_count = len(detected_hand_tiles)
        self._recent_hand_counts.append(hand_count)
        hand_stable = len(self._recent_hand_counts) >= self._stable_frames and len(set(self._recent_hand_counts)) == 1
        corrected = False
        is_self_turn_hand = False
        if hand_stable:
            self._stable_hand_count = hand_count
            is_self_turn_hand = is_self_discard_turn_by_hand_count(detected_hand_tiles)
            if not is_self_turn_hand:
                self._self_discard_turn_active = False

        table_events: List[Dict[str, Any]] = []
        table_obs = self._normalize_table_observations(table_observations)
        table_signature = self._table_signature(table_obs)
        table_stable = False

        if table_signature == self._table_candidate_signature:
            self._table_candidate_frames += 1
            self._table_candidate_obs = table_obs
        else:
            self._table_candidate_signature = table_signature
            self._table_candidate_frames = 1
            self._table_candidate_obs = table_obs

        if self._table_candidate_frames >= self._table_stable_frames:
            table_stable = True
            prev_obs = self._stable_table_obs
            cur_obs = list(self._table_candidate_obs)
            if self._reset_table_baseline_on_next_stable:
                self._stable_table_signature = table_signature
                self._stable_table_obs = cur_obs
                self._reset_table_baseline_on_next_stable = False
                table_events.append(
                    {
                        "type": "table_baseline_reset",
                        "count": len(cur_obs),
                    }
                )
            elif self._stable_table_signature != table_signature:
                if self._pending_self_table_tile is not None:
                    if self._stable_table_signature is None:
                        self._stable_table_signature = table_signature
                        self._stable_table_obs = cur_obs
                        expected_tile = self._pending_self_table_tile
                        self._pending_self_table_tile = None
                        table_events.append(
                            {
                                "type": "table_self_discard_baseline",
                                "tile_id": int(expected_tile) if expected_tile is not None else -1,
                                "tile": tile_id_to_label(int(expected_tile)) if expected_tile is not None else "",
                                "matched": any(
                                    int(obs.get("tile_id", -1)) == int(expected_tile)
                                    for obs in cur_obs
                                )
                                if expected_tile is not None
                                else False,
                                "count": len(cur_obs),
                            }
                        )
                    elif len(cur_obs) == len(prev_obs) + 1:
                        new_obs = self._find_new_table_observations(prev_obs, cur_obs)
                        if new_obs:
                            table_events.append(self._accept_pending_self_table_baseline(cur_obs, new_obs))
                        else:
                            table_events.append(
                                {
                                    "type": "table_new_tile_unmatched",
                                    "previous_count": len(prev_obs),
                                    "current_count": len(cur_obs),
                                }
                            )
                    elif len(cur_obs) == len(prev_obs):
                        table_events.append(
                            {
                                "type": "table_waiting_self_discard_baseline",
                                "count": len(cur_obs),
                            }
                        )
                    elif len(cur_obs) > len(prev_obs) + 1:
                        table_events.append(
                            {
                                "type": "table_count_jump_ignored",
                                "previous_count": len(prev_obs),
                                "current_count": len(cur_obs),
                            }
                        )
                    else:
                        table_events.append(
                            {
                                "type": "table_count_drop_ignored",
                                "previous_count": len(prev_obs),
                                "current_count": len(cur_obs),
                            }
                        )
                elif self._stable_table_signature is None:
                    self._stable_table_signature = table_signature
                    self._stable_table_obs = cur_obs
                    table_events.append(
                        {
                            "type": "table_baseline_stable",
                            "count": len(cur_obs),
                        }
                    )
                elif len(cur_obs) == len(prev_obs) + 1:
                    new_obs = self._find_new_table_observations(prev_obs, cur_obs)
                    if new_obs:
                        if self._expected_discarder == 0 and self._self_discard_turn_active:
                            table_events.append(
                                {
                                    "type": "table_waiting_self_discard_hand_exit",
                                    "previous_count": len(prev_obs),
                                    "current_count": len(cur_obs),
                                }
                            )
                        else:
                            table_events.append(self._accept_new_discard(new_obs[0]))
                            self._stable_table_signature = table_signature
                            self._stable_table_obs = cur_obs
                    else:
                        table_events.append(
                            {
                                "type": "table_new_tile_unmatched",
                                "previous_count": len(prev_obs),
                                "current_count": len(cur_obs),
                            }
                        )
                elif len(cur_obs) == len(prev_obs):
                    table_events.append(
                        {
                            "type": "table_same_count_stable",
                            "count": len(prev_obs),
                        }
                    )
                elif len(cur_obs) > len(prev_obs) + 1:
                    table_events.append(
                        {
                            "type": "table_count_jump_ignored",
                            "previous_count": len(prev_obs),
                            "current_count": len(cur_obs),
                        }
                    )
                else:
                    table_events.append(
                        {
                            "type": "table_count_drop_ignored",
                            "previous_count": len(prev_obs),
                            "current_count": len(cur_obs),
                        }
                    )

        if hand_stable and is_self_turn_hand and not self._self_discard_turn_active:
            self.correct_to_self_discard()
            corrected = True

        info = {
            "corrected": corrected,
            "hand_stable": hand_stable,
            "stable_count_frames": len(self._recent_hand_counts),
            "stable_hand_count": self._stable_hand_count if hand_stable else None,
            "table_stable": table_stable,
            "table_candidate_frames": self._table_candidate_frames,
            "table_stable_required_frames": self._table_stable_frames,
            "stable_table_count": len(self._stable_table_obs),
            "table_events": table_events,
            "expected_discarder": self._expected_discarder,
        }
        self._last_live_info = info
        return info


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


def _can_pon(hand_tiles: Sequence[int], tile_id: Optional[int]) -> bool:
    if tile_id is None:
        return False
    return list(int(t) for t in hand_tiles).count(int(tile_id)) >= 2


def _chi_variants(hand_tiles: Sequence[int], tile_id: Optional[int], discarder: Optional[int]) -> Dict[int, Tuple[int, int]]:
    variants: Dict[int, Tuple[int, int]] = {}
    if tile_id is None or discarder != 3 or int(tile_id) >= 27:
        return variants
    tile = int(tile_id)
    hand = [int(t) for t in hand_tiles]
    suit = tile // 9
    idx = tile % 9
    base = suit * 9

    candidates = (
        (ACT_CHI_LOW, idx >= 2, base + idx - 2, base + idx - 1),
        (ACT_CHI_MID, 1 <= idx <= 7, base + idx - 1, base + idx + 1),
        (ACT_CHI_HIGH, idx <= 6, base + idx + 1, base + idx + 2),
    )
    for action, ok, a, b in candidates:
        if ok and hand.count(a) >= 1 and hand.count(b) >= 1:
            variants[action] = (a, b)
    return variants


def legal_action_mask(tracker: GameStateTracker, hand_tiles: Sequence[int]) -> np.ndarray:
    mask = np.zeros((ACTION_DIM,), dtype=bool)
    clean_hand = [int(t) for t in hand_tiles if 0 <= int(t) < 34]

    if tracker.phase == PHASE_CLAIM and tracker.last_discarder != 0:
        mask[ACT_PASS] = True
        if _can_pon(clean_hand, tracker.last_discard):
            mask[ACT_PON] = True
        for action in _chi_variants(clean_hand, tracker.last_discard, tracker.last_discarder):
            mask[action] = True
        return mask

    if tracker.phase == PHASE_DISCARD and tracker.current_player == 0:
        for tile in clean_hand:
            mask[int(tile)] = True
        if not mask.any():
            mask[ACT_PASS] = True
        return mask

    mask[ACT_PASS] = True
    return mask


def _is_maskable_model(rl_model) -> bool:
    if rl_model is None:
        return False
    name = type(rl_model).__name__.lower()
    module = type(rl_model).__module__.lower()
    return "maskable" in name or "sb3_contrib" in module


def _predict_policy_action(rl_model, obs: np.ndarray, mask: np.ndarray) -> int:
    if _is_maskable_model(rl_model):
        action, _ = rl_model.predict(obs, deterministic=True, action_masks=mask)
    else:
        action, _ = rl_model.predict(obs, deterministic=True)
    return int(action)


def _topk_ppo_action_probs(
    rl_model,
    obs: np.ndarray,
    top_k: int = 5,
    action_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    if rl_model is None or torch is None or top_k <= 0:
        return []
    try:
        device = getattr(rl_model, "device", "cpu")
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if _is_maskable_model(rl_model):
            dist = rl_model.policy.get_distribution(obs_t, action_masks=action_mask)
        else:
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


def _empty_live_agent_result(
    detected_hand_tiles: Sequence[int],
    reason: str,
    tracker: Optional[GameStateTracker] = None,
    live_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "corrected": False,
        "current_player": tracker.current_player if tracker is not None else 0,
        "phase": tracker.phase if tracker is not None else PHASE_DISCARD,
        "stable_count_frames": 0,
        "hand_count": len(detected_hand_tiles),
        "recommended_action": -1,
        "recommended_tile": "",
        "decision_type": "none",
        "source": "none",
        "reason": reason,
        "topk": [],
        "live": live_info or {},
    }


def maybe_correct_self_turn_and_suggest(
    tracker: GameStateTracker,
    detected_hand_tiles: Sequence[int],
    table_observations: Optional[Sequence[Dict[str, Any]]] = None,
    rl_model=None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Debounce live vision, update turn state, and ask the PPO policy when it is our decision."""
    hand_tiles = [int(t) for t in detected_hand_tiles if 0 <= int(t) < 34]
    live_info = tracker.update_live_state(hand_tiles, table_observations)
    action_mask = legal_action_mask(tracker, hand_tiles)

    if not live_info.get("hand_stable", False):
        result = _empty_live_agent_result(hand_tiles, "hand count is not stable yet", tracker, live_info)
        result["stable_count_frames"] = int(live_info.get("stable_count_frames", 0))
        return result

    decision_type = "none"
    if tracker.phase == PHASE_DISCARD and tracker.current_player == 0:
        if not is_self_discard_turn_by_hand_count(hand_tiles):
            result = _empty_live_agent_result(hand_tiles, "stable hand count is not 3n+2", tracker, live_info)
            result["stable_count_frames"] = int(live_info.get("stable_count_frames", 0))
            return result
        decision_type = "discard"
    elif tracker.phase == PHASE_CLAIM and tracker.last_discarder != 0:
        decision_type = "claim"
    else:
        result = _empty_live_agent_result(hand_tiles, "waiting for another player's discard", tracker, live_info)
        result["stable_count_frames"] = int(live_info.get("stable_count_frames", 0))
        return result

    obs = build_obs_from_tracker(tracker, hand_tiles)
    source = "ppo"
    reason = (
        "PPO policy (deterministic) for discard"
        if decision_type == "discard"
        else "PPO policy (deterministic) for claim/pass"
    )
    action_id: Optional[int] = None

    if rl_model is not None:
        try:
            action_id = _predict_policy_action(rl_model, obs, action_mask)
        except Exception as e:
            source = "fallback"
            if decision_type == "discard":
                reason = f"PPO predict failed; fallback discard ({e})"
                action_id = fallback_discard_tile(hand_tiles)
            else:
                reason = f"PPO predict failed; fallback PASS ({e})"
                action_id = ACT_PASS
    else:
        source = "fallback"
        if decision_type == "discard":
            reason = "PPO model not loaded; fallback discard"
            action_id = fallback_discard_tile(hand_tiles)
        else:
            reason = "PPO model not loaded; fallback PASS"
            action_id = ACT_PASS

    if action_id is None or not (0 <= int(action_id) < ACTION_DIM) or not bool(action_mask[int(action_id)]):
        source = "fallback" if source == "ppo" else source
        if decision_type == "discard":
            reason = "PPO action illegal for detected hand; fallback discard"
            action_id = fallback_discard_tile(hand_tiles)
        else:
            reason = "PPO action illegal for claim state; fallback PASS"
            action_id = ACT_PASS

    if decision_type == "discard":
        tile = tile_id_to_label(action_id) if action_id is not None and 0 <= int(action_id) < 34 else ""
    else:
        tile = action_to_label(int(action_id)) if action_id is not None else ""

    return {
        "corrected": bool(live_info.get("corrected", False)),
        "current_player": tracker.current_player,
        "phase": tracker.phase,
        "stable_count_frames": int(live_info.get("stable_count_frames", 0)),
        "cooldown_frames_remaining": tracker.cooldown_frames_remaining,
        "hand_count": len(hand_tiles),
        "last_discard": tracker.last_discard,
        "last_discard_tile": tile_id_to_label(tracker.last_discard) if tracker.last_discard is not None else "",
        "last_discarder": tracker.last_discarder,
        "recommended_action": int(action_id) if action_id is not None else -1,
        "recommended_tile": tile,
        "decision_type": decision_type,
        "source": source,
        "reason": reason,
        "action_mask": [bool(x) for x in action_mask.tolist()],
        "topk": _topk_ppo_action_probs(rl_model, obs, top_k=top_k, action_mask=action_mask),
        "live": live_info,
    }


def load_ppo_model(ppo_model_path: Optional[str], ppo_device: Optional[str] = None):
    """Load SB3 PPO/MaskablePPO model if available. Returns model or None."""
    if not ppo_model_path:
        return None
    if PPO is None and MaskablePPO is None:
        print("[PPO] stable-baselines3/sb3-contrib not installed.")
        return None

    try:
        if ppo_device is not None:
            dev = ppo_device
        else:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
        load_errors = []
        if MaskablePPO is not None:
            try:
                model = MaskablePPO.load(ppo_model_path, device=dev)
                print(f"[PPO] Loaded MaskablePPO model: {ppo_model_path} (device={dev})")
                return model
            except Exception as e:
                load_errors.append(f"MaskablePPO: {e}")
        if PPO is not None:
            try:
                model = PPO.load(ppo_model_path, device=dev)
                print(f"[PPO] Loaded PPO model: {ppo_model_path} (device={dev})")
                return model
            except Exception as e:
                load_errors.append(f"PPO: {e}")
        raise RuntimeError("; ".join(load_errors))
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
