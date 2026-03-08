# tw16mj_gym_env_hand34_claim.py
# 台灣 16 張麻將 - 低維 + 吃碰（Chi/Pon）決策版本
#
# ✅ Observation（較低維，但包含你要求的河牌資訊）：
#   - 自己手牌 + pending draw：34 維計數
#   - 四家河牌（打出過的牌）counts：4 * 34
#   - last_discard / last_discarder / phase：用 3 個 scalar 編碼（降低維度）
#   - 自己已吃/碰形成的 open meld 計數（順子/刻子各一個 scalar）
#   => obs_dim = 34 + 136 + 3 + 2 = 175
#
# ✅ Action（單一 Discrete，含出牌 + 吃碰）：
#   - 0..33：丟牌值（DISCARD phase 才有效）
#   - 34：PASS（CLAIM phase）
#   - 35：PON（碰）（CLAIM phase）
#   - 36：CHI_LOW  （吃：x-2, x-1, x）只允許吃上家（player3）
#   - 37：CHI_MID  （吃：x-1, x, x+1）只允許吃上家
#   - 38：CHI_HIGH （吃：x, x+1, x+2）只允許吃上家
#
# ✅ Reward（保持與 hand34_4step 相同機制，不變）：
#   - 每步小懲罰
#   - 打牌/吃碰後進聽數（4-step）減少：獎勵（依幅度）
#   - 打牌/吃碰後進聽數增加：懲罰（依幅度）
#   - 胡牌大獎勵
#
# ⚠️ 簡化假設（目前版本）：
#   - 只有 Player0（你）會做吃/碰；NPC 仍只摸打（heuristic 丟牌）
#   - 胡牌判定沿用 hu_result.hu(hand, tile)
#   - 你對別人打出的牌若可胡：視為「自動胡」（不再多一個 action）

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    import gym
    from gym import spaces

import hu_result
from mj_utils import next_not_block, next_two_not_blsame


N_TILE_TYPES = 34
FLOWER_START, FLOWER_END = 34, 42
START_HAND = 16
PLAYERS = 4
TOTAL_TILES = 144

PHASE_DISCARD = 0
PHASE_CLAIM = 1

ACT_PASS = 34
ACT_PON = 35
ACT_CHI_LOW = 36
ACT_CHI_MID = 37
ACT_CHI_HIGH = 38

STEP_PENALTY = -0.05
REWARD_PROGRESS_DEC = 1.0
PENALTY_PROGRESS_INC = -1.0
WIN_REWARD_SELF = 15.0
LOSE_PENALTY = -5.0
INVALID_ACTION_PENALTY = -0.10

DEFAULT_MAX_STEPS = 500


def build_wall(rng: random.Random) -> List[int]:
    wall: List[int] = []
    for t in range(N_TILE_TYPES):
        wall.extend([t] * 4)
    wall.extend(list(range(FLOWER_START, FLOWER_END)))
    rng.shuffle(wall)
    return wall


def draw_one(wall: List[int]) -> Optional[int]:
    if not wall:
        return None
    return wall.pop()


def resolve_flowers(wall: List[int], flowers: List[int], first_draw: Optional[int]) -> Optional[int]:
    t = first_draw
    while t is not None and FLOWER_START <= t < FLOWER_END:
        flowers.append(t)
        t = draw_one(wall)
    return t


def insert_sorted(hand: List[int], tile: int) -> List[int]:
    lo, hi = 0, len(hand)
    while lo < hi:
        mid = (lo + hi) // 2
        if tile <= hand[mid]:
            hi = mid
        else:
            lo = mid + 1
    return hand[:lo] + [tile] + hand[lo:]


def remove_k(hand: List[int], tile: int, k: int) -> bool:
    need = k
    i = 0
    while i < len(hand) and need > 0:
        if hand[i] == tile:
            del hand[i]
            need -= 1
            continue
        i += 1
    return need == 0


def count34_from_list(lst: List[int]) -> np.ndarray:
    c = np.zeros((N_TILE_TYPES,), dtype=np.float32)
    for v in lst:
        if 0 <= v < N_TILE_TYPES:
            c[v] += 1.0
    return c


def hand_pending_count34(hand: List[int], pending: Optional[int]) -> np.ndarray:
    c = count34_from_list(hand)
    if pending is not None and 0 <= pending < N_TILE_TYPES:
        c[int(pending)] += 1.0
    return c


# ---- heuristic discard ----
def same3_block(mj: List[int], block: List[int]) -> List[int]:
    i = 0
    mj_num = len(mj)
    while i < mj_num - 2:
        if block[i] == 1:
            i += 1
        elif mj[i] == mj[i + 1] == mj[i + 2]:
            block[i] = block[i + 1] = block[i + 2] = 1
            i += 3
        else:
            i += 1
    return block


def seq3_block(mj: List[int], block: List[int]) -> List[int]:
    i = 0
    mj_num = len(mj)
    while i < mj_num - 2:
        if block[i] == 1:
            i += 1
            continue
        m1, m2 = next_two_not_blsame(block, mj_num, i + 1, mj, mj[i])
        if m1 != -1 and m2 != -1:
            if (
                mj[i] < 27
                and mj[i] // 9 == mj[m1] // 9 == mj[m2] // 9
                and mj[i] + 1 == mj[m1]
                and mj[m1] + 1 == mj[m2]
            ):
                block[i] = block[m1] = block[m2] = 1
        i += 1
    return block


def add_block3(mj: List[int], block: List[int]) -> List[int]:
    block = same3_block(mj, block)
    block = seq3_block(mj, block)
    return block


def add_block2(mj: List[int], block: List[int]) -> List[int]:
    mj_num = len(mj)
    i = 0
    while block.count(0) > 2 and i < mj_num - 1:
        if block[i] == 0:
            j = next_not_block(block, mj_num, i + 1)
            if j != -1:
                if mj[i] == mj[j]:
                    block[i] = block[j] = 1
                elif mj[i] < 27 and mj[i] // 9 == mj[j] // 9 and mj[i] + 1 == mj[j]:
                    block[i] = block[j] = 1
        i += 1
    return block


def heuristic_discard_tile(hand: List[int]) -> int:
    tmj = sorted(hand)
    block = [0] * len(tmj)
    block = add_block3(tmj, block)
    if block.count(0) > 2:
        block = add_block2(tmj, block)
    di = next_not_block(block, len(block), 0)
    return tmj[-1] if di == -1 else tmj[di]


# ---- shanten/tenpai steps (4-step) with open melds ----
def _counts_from_hand(hand: List[int]) -> List[int]:
    c = [0] * N_TILE_TYPES
    for v in hand:
        if 0 <= v < N_TILE_TYPES:
            c[v] += 1
    return c


def tenpai_steps_4step(hand_tiles: List[int], open_n3c: int = 0, open_n3s: int = 0) -> int:
    cnt = _counts_from_hand(hand_tiles)

    # Step1: sequence sets
    N3C = int(open_n3c)
    for base in (0, 9, 18):
        for i in range(0, 7):
            m = min(cnt[base + i], cnt[base + i + 1], cnt[base + i + 2])
            if m > 0:
                N3C += m
                cnt[base + i] -= m
                cnt[base + i + 1] -= m
                cnt[base + i + 2] -= m

    # Step2: triplets
    N3S = int(open_n3s)
    for t in range(N_TILE_TYPES):
        m = cnt[t] // 3
        if m > 0:
            N3S += m
            cnt[t] -= 3 * m

    # Step3: pairs
    N2S = 0
    for t in range(N_TILE_TYPES):
        m = cnt[t] // 2
        if m > 0:
            N2S += m
            cnt[t] -= 2 * m

    # Step4: +/-2 connections (numbers only)
    N2C = 0
    for base in (0, 9, 18):
        for i in range(0, 9):
            if i + 1 <= 8:
                m = min(cnt[base + i], cnt[base + i + 1])
                if m > 0:
                    N2C += m
                    cnt[base + i] -= m
                    cnt[base + i + 1] -= m
            if i + 2 <= 8:
                m = min(cnt[base + i], cnt[base + i + 2])
                if m > 0:
                    N2C += m
                    cnt[base + i] -= m
                    cnt[base + i + 2] -= m

    missing = 5 - N3S - N3C - N2S - N2C
    if missing < 0:
        missing = 0
    steps = missing * 2 + N2C + N2S
    if steps < 0:
        steps = 0
    return int(steps)


@dataclass
class Meld:
    kind: str  # "CHI" or "PON"
    tiles: Tuple[int, int, int]
    taken: int


def meld_counts(melds: List[Meld]) -> Tuple[int, int]:
    n3c = 0
    n3s = 0
    for m in melds:
        if m.kind == "PON":
            n3s += 1
        elif m.kind == "CHI":
            n3c += 1
    return n3c, n3s


class Tw16MahjongEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None, max_steps: int = DEFAULT_MAX_STEPS):
        super().__init__()
        self.rng = random.Random(seed)
        self.max_steps = max_steps

        obs_dim = 34 + (4 * 34) + 3 + 2  # 175
        self.observation_space = spaces.Box(low=0.0, high=4.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(39)

        self.wall: List[int] = []
        self.hands: List[List[int]] = [[] for _ in range(PLAYERS)]
        self.flowers: List[List[int]] = [[] for _ in range(PLAYERS)]
        self.discards: List[List[int]] = [[] for _ in range(PLAYERS)]
        self.melds: List[List[Meld]] = [[] for _ in range(PLAYERS)]

        self.pending_draw: Optional[int] = None
        self.phase: int = PHASE_DISCARD
        self.next_player: int = 0

        self.last_discard: Optional[int] = None
        self.last_discarder: Optional[int] = None

        self.steps: int = 0
        self.done: bool = False

    def _draw_nonflower_for_player(self, pid: int) -> Optional[int]:
        return resolve_flowers(self.wall, self.flowers[pid], draw_one(self.wall))

    def _obs(self) -> np.ndarray:
        v_hand = hand_pending_count34(self.hands[0], self.pending_draw)
        v_discards = np.concatenate([count34_from_list(self.discards[p]) for p in range(PLAYERS)], axis=0)

        ld = 0.0 if self.last_discard is None else (float(self.last_discard) + 1.0) / 34.0
        ldp = 0.0 if self.last_discarder is None else (float(self.last_discarder) + 1.0) / 4.0
        ph = float(self.phase)
        v_scalars = np.array([ld, ldp, ph], dtype=np.float32)

        open_n3c, open_n3s = meld_counts(self.melds[0])
        v_open = np.array([open_n3c / 5.0, open_n3s / 5.0], dtype=np.float32)

        return np.concatenate([v_hand, v_discards, v_scalars, v_open], axis=0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng.seed(seed)

        self.wall = build_wall(self.rng)
        self.hands = [[] for _ in range(PLAYERS)]
        self.flowers = [[] for _ in range(PLAYERS)]
        self.discards = [[] for _ in range(PLAYERS)]
        self.melds = [[] for _ in range(PLAYERS)]

        self.pending_draw = None
        self.phase = PHASE_DISCARD
        self.next_player = 0
        self.last_discard = None
        self.last_discarder = None
        self.steps = 0
        self.done = False

        for p in range(PLAYERS):
            while len(self.hands[p]) < START_HAND:
                t = resolve_flowers(self.wall, self.flowers[p], draw_one(self.wall))
                if t is None:
                    break
                self.hands[p].append(int(t))
            self.hands[p].sort()

        self.pending_draw = self._draw_nonflower_for_player(0)

        retry = 6
        while retry > 0 and self.pending_draw is not None and hu_result.hu(self.hands[0], int(self.pending_draw)) == 1:
            self.wall.insert(0, int(self.pending_draw))
            self.pending_draw = self._draw_nonflower_for_player(0)
            retry -= 1

        return self._obs(), {}

    # ---- claim helpers ----
    def _can_pon(self, tile: int) -> bool:
        return self.hands[0].count(tile) >= 2

    def _can_chi_variants(self, tile: int, discarder: int) -> Dict[int, Tuple[int, int]]:
        res: Dict[int, Tuple[int, int]] = {}
        if discarder != 3:
            return res
        if tile >= 27:
            return res
        suit = tile // 9
        idx = tile % 9
        base = suit * 9

        if idx >= 2:
            a, b = base + (idx - 2), base + (idx - 1)
            if self.hands[0].count(a) >= 1 and self.hands[0].count(b) >= 1:
                res[ACT_CHI_LOW] = (a, b)
        if 1 <= idx <= 7:
            a, b = base + (idx - 1), base + (idx + 1)
            if self.hands[0].count(a) >= 1 and self.hands[0].count(b) >= 1:
                res[ACT_CHI_MID] = (a, b)
        if idx <= 6:
            a, b = base + (idx + 1), base + (idx + 2)
            if self.hands[0].count(a) >= 1 and self.hands[0].count(b) >= 1:
                res[ACT_CHI_HIGH] = (a, b)

        return res

    def _claim_options_exist(self, tile: int, discarder: int) -> bool:
        return self._can_pon(tile) or (len(self._can_chi_variants(tile, discarder)) > 0)

    def _apply_claim(self, action: int) -> Tuple[bool, float]:
        if self.last_discard is None or self.last_discarder is None:
            return False, 0.0
        tile = int(self.last_discard)
        discarder = int(self.last_discarder)

        open_n3c, open_n3s = meld_counts(self.melds[0])
        old_prog = tenpai_steps_4step(self.hands[0], open_n3c, open_n3s)

        claimed = False
        if action == ACT_PON:
            if self._can_pon(tile) and remove_k(self.hands[0], tile, 2):
                self.melds[0].append(Meld(kind="PON", tiles=(tile, tile, tile), taken=tile))
                claimed = True
        elif action in (ACT_CHI_LOW, ACT_CHI_MID, ACT_CHI_HIGH):
            chi = self._can_chi_variants(tile, discarder)
            if action in chi:
                a, b = chi[action]
                if remove_k(self.hands[0], a, 1) and remove_k(self.hands[0], b, 1):
                    self.melds[0].append(Meld(kind="CHI", tiles=(a, b, tile), taken=tile))
                    claimed = True

        delta = 0.0
        if claimed:
            if self.discards[discarder] and self.discards[discarder][-1] == tile:
                self.discards[discarder].pop()

            open_n3c2, open_n3s2 = meld_counts(self.melds[0])
            new_prog = tenpai_steps_4step(self.hands[0], open_n3c2, open_n3s2)
            if new_prog < old_prog:
                delta += REWARD_PROGRESS_DEC * (old_prog - new_prog)
            elif new_prog > old_prog:
                delta += PENALTY_PROGRESS_INC * (new_prog - old_prog)

        return claimed, delta

    # ---- runner ----
    def _run_until_agent_decision(self, reward: float, info: Dict):
        while True:
            if self.steps >= self.max_steps:
                self.done = True
                info["truncated"] = True
                return self._obs(), float(reward), False, True, info

            p = int(self.next_player)
            if p == 0:
                self.pending_draw = self._draw_nonflower_for_player(0)
                if self.pending_draw is None:
                    self.done = True
                    info["draw"] = True
                    return self._obs(), float(reward), True, False, info
                if hu_result.hu(self.hands[0], int(self.pending_draw)) == 1:
                    self.done = True
                    info["winner"] = 0
                    reward += WIN_REWARD_SELF
                    return self._obs(), float(reward), True, False, info

                self.phase = PHASE_DISCARD
                return self._obs(), float(reward), False, False, info

            t = self._draw_nonflower_for_player(p)
            if t is None:
                self.done = True
                info["draw"] = True
                return self._obs(), float(reward), True, False, info

            if hu_result.hu(self.hands[p], int(t)) == 1:
                self.done = True
                info["winner"] = p
                reward += LOSE_PENALTY
                return self._obs(), float(reward), True, False, info

            self.hands[p] = insert_sorted(self.hands[p], int(t))
            dv = heuristic_discard_tile(self.hands[p])
            remove_k(self.hands[p], dv, 1)
            self.discards[p].append(dv)

            self.last_discard = dv
            self.last_discarder = p

            if hu_result.hu(self.hands[0], dv) == 1:
                self.done = True
                info["winner"] = 0
                reward += WIN_REWARD_SELF
                return self._obs(), float(reward), True, False, info

            if self._claim_options_exist(dv, p):
                self.phase = PHASE_CLAIM
                self.pending_draw = None
                self.next_player = (p + 1) % 4
                return self._obs(), float(reward), False, False, info

            self.next_player = (p + 1) % 4

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Env is done. Call reset().")

        self.steps += 1
        reward = STEP_PENALTY
        info: Dict = {}

        if self.phase == PHASE_CLAIM:
            if action not in (ACT_PASS, ACT_PON, ACT_CHI_LOW, ACT_CHI_MID, ACT_CHI_HIGH):
                reward += INVALID_ACTION_PENALTY
                action = ACT_PASS

            if action == ACT_PASS:
                self.phase = PHASE_DISCARD
                return self._run_until_agent_decision(reward, info)

            claimed, delta = self._apply_claim(action)
            if not claimed:
                reward += INVALID_ACTION_PENALTY
                self.phase = PHASE_DISCARD
                return self._run_until_agent_decision(reward, info)

            reward += delta
            self.phase = PHASE_DISCARD
            self.pending_draw = None
            self.next_player = 0
            return self._obs(), float(reward), False, False, info

        # DISCARD
        if self.next_player != 0:
            return self._run_until_agent_decision(reward, info)

        open_n3c, open_n3s = meld_counts(self.melds[0])
        old_prog = tenpai_steps_4step(self.hands[0], open_n3c, open_n3s)

        avail = list(self.hands[0])
        if self.pending_draw is not None:
            avail.append(int(self.pending_draw))

        if not (0 <= action <= 33):
            reward += INVALID_ACTION_PENALTY
            discard = heuristic_discard_tile(avail)
        else:
            discard = int(action)
            if discard not in avail:
                reward += INVALID_ACTION_PENALTY
                discard = heuristic_discard_tile(avail)

        if self.pending_draw is not None:
            pd = int(self.pending_draw)
            if discard == pd:
                self.pending_draw = None
            else:
                remove_k(self.hands[0], discard, 1)
                self.hands[0] = insert_sorted(self.hands[0], pd)
                self.pending_draw = None
        else:
            remove_k(self.hands[0], discard, 1)

        self.discards[0].append(discard)
        self.last_discard = discard
        self.last_discarder = 0

        open_n3c2, open_n3s2 = meld_counts(self.melds[0])
        new_prog = tenpai_steps_4step(self.hands[0], open_n3c2, open_n3s2)
        if new_prog < old_prog:
            reward += REWARD_PROGRESS_DEC * (old_prog - new_prog)
        elif new_prog > old_prog:
            reward += PENALTY_PROGRESS_INC * (new_prog - old_prog)

        for p in (1, 2, 3):
            if hu_result.hu(self.hands[p], discard) == 1:
                self.done = True
                info["winner"] = p
                reward += LOSE_PENALTY
                return self._obs(), float(reward), True, False, info

        self.next_player = 1
        return self._run_until_agent_decision(reward, info)
