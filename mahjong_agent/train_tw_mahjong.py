# train_tw_mahjong.py
# Minimal RL training scaffold for "16-tile Taiwan Mahjong" with Player0 as the learning agent.
# Other players (1,2,3) use fixed heuristic logic derived from your code structure.

import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ============= 基本常數（對齊你的索引約定） =============
N_TILE_TYPES = 34             # 0..33：萬/條/筒/字牌（不含花）
FLOWER_START, FLOWER_END = 34, 42  # 34..41 花牌（本骨架先略過花牌細節）
START_HAND = 16               # 台麻 16 張
PLAYERS = 4

# ------------------------------------------------------
# 工具：對齊你原本邏輯的「組合與丟牌」啟發式（簡化版）
# ------------------------------------------------------
def next_not_block(block: List[int], mj_num: int, start: int = 0) -> int:
    i = start
    while i < mj_num:
        if block[i] == 0:
            return i
        i += 1
    return -1

def next_two_not_block(block: List[int], mj_num: int, start: int = 0) -> Tuple[int, int]:
    n0 = next_not_block(block, mj_num, start)
    if n0 == -1: return -1, -1
    n1 = next_not_block(block, mj_num, n0 + 1)
    if n1 == -1: return n0, -1
    return n0, n1

def next_not_blsame(block: List[int], mj_num: int, mj: List[int], same_val: int, start: int = 0) -> int:
    i = start
    while i < mj_num:
        if block[i] == 0 and mj[i] != same_val:
            return i
        i += 1
    return -1

def same3_block(mj: List[int], mj_num: int, block: List[int]) -> List[int]:
    i = 0
    while i < mj_num - 2:
        if block[i] == 1:
            i += 1
        elif mj[i] == mj[i+1] == mj[i+2]:
            block[i] = block[i+1] = block[i+2] = 1
            i += 3
        else:
            i += 1
    return block

def seq3_block(mj: List[int], mj_num: int, block: List[int]) -> List[int]:
    i = 0
    while i < mj_num - 2:
        if block[i] == 1:
            i += 1
            continue
        m1 = next_not_blsame(block, mj_num, mj, mj[i], i+1)
        if m1 == -1:
            i += 1; continue
        m2 = next_not_blsame(block, mj_num, mj, mj[i], m1+1)
        if m2 == -1:
            i += 1; continue
        # 只有序數牌能組順子，且必須同花色連號
        if mj[i] < 27 and (mj[i] // 9 == mj[m1] // 9 == mj[m2] // 9) and (mj[i] + 1 == mj[m1]) and (mj[m1] + 1 == mj[m2]):
            block[i] = block[m1] = block[m2] = 1
        i += 1
    return block

def add_block3(mj: List[int], block: List[int]) -> List[int]:
    mj_num = len(mj)
    block = same3_block(mj, mj_num, block)
    block = seq3_block(mj, mj_num, block)
    return block

def add_block2(mj: List[int], block: List[int]) -> List[int]:
    # 嘗試配出對子或兩張連牌（用於「快完成」的近似）
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

def heuristic_discard(mj: List[int]) -> int:
    """丟出第一張未參與任何3張（刻/順）或2張（對/連）組合的牌。回傳該牌值。"""
    tmj = sorted(mj[:])
    block = [0] * len(tmj)
    block = add_block3(tmj, block)
    if block.count(0) > 2:
        block = add_block2(tmj, block)
    di = next_not_block(block, len(block))
    if di == -1:
        # 理論上不會發生；保底丟最右邊
        return tmj[-1]
    return tmj[di]

def dark_kong_available(mj: List[int]) -> Optional[int]:
    """有沒有四張相同（暗槓）。回傳該牌值或 None。"""
    tmj = sorted(mj)
    for v in range(N_TILE_TYPES):
        if tmj.count(v) >= 4:
            return v
    return None

def count_structure_for_reward(mj: List[int]) -> Tuple[int, int, int, int]:
    """
    粗略統計出牌後的結構：
    回傳 (n_set3, n_pair, n_adj, n_single)
      n_set3 : 3張組(刻/順)的總數
      n_pair : 對子數（同牌兩張）
      n_adj  : 相鄰兩張數（同花色連號兩張）
      n_single: 剩餘孤張數
    註：這是「獎勵用啟發式」，不是嚴格胡牌判定。
    """
    cnt = [0] * N_TILE_TYPES
    for v in mj:
        if 0 <= v < N_TILE_TYPES:
            cnt[v] += 1

    # 1) 先拿掉所有「刻」
    n_pung = 0
    for t in range(N_TILE_TYPES):
        k = cnt[t] // 3
        if k > 0:
            n_pung += k
            cnt[t] -= 3 * k

    # 2) 再拿「順」
    n_chow = 0
    for base in (0, 9, 18):  # 萬、條、筒
        for i in range(0, 7):
            m = min(cnt[base + i], cnt[base + i + 1], cnt[base + i + 2])
            if m > 0:
                n_chow += m
                cnt[base + i]     -= m
                cnt[base + i + 1] -= m
                cnt[base + i + 2] -= m

    n_set3 = n_pung + n_chow

    # 3) 對子
    n_pair = 0
    for t in range(N_TILE_TYPES):
        p = cnt[t] // 2
        if p > 0:
            n_pair += p
            cnt[t] -= 2 * p

    # 4) 相鄰兩張（同花色 i, i+1）
    n_adj = 0
    for base in (0, 9, 18):
        for i in range(0, 8):
            m = min(cnt[base + i], cnt[base + i + 1])
            if m > 0:
                n_adj += m
                cnt[base + i]     -= m
                cnt[base + i + 1] -= m

    # 5) 孤張
    n_single = sum(cnt)
    return n_set3, n_pair, n_adj, n_single

# ------------------------------------------------------
# 向聽數（近似）：16 張台麻需「5 面子 + 1 對子」
# ------------------------------------------------------
def approx_shanten_16(mj: List[int]) -> int:
    """
    近似向聽數（數字愈小愈好；=0 表示聽牌）。
    估法：先封 3張組合，再封 2張組合；計算距離 5 面子 + 1 對子 的欠缺量。
    """
    tmj = sorted(mj[:])
    block = [0] * len(tmj)

    # 優先找 3 張組合
    block = add_block3(tmj, block)

    # 數已完成的「刻/順」
    used3 = 0
    i = 0
    while i < len(tmj):
        if block[i] == 1:
            used3 += 1
            skip = 1
            c = i + 1
            while c < len(tmj) and skip < 3 and block[c] == 1:
                skip += 1; c += 1
            i = c
        else:
            i += 1

    # 再封 2 張組合（對/連）
    block = add_block2(tmj, block)
    used2 = 0
    i = 0
    while i < len(tmj) - 1:
        if block[i] == 1 and block[i+1] == 1:
            used2 += 1
            i += 2
        else:
            i += 1

    need_sets = max(0, 5 - used3)  # 5 面子
    need_pair = 1                  # 1 對子
    rem2 = max(0, need_sets + need_pair - used2)
    shanten = rem2
    return max(0, shanten)

# ---------- 幾進聽 & 孤張判定（全域函式，子進程可引用） ----------
def steps_to_tenpai(mj: List[int]) -> int:
    """以近似向聽作為『最少摸打幾次可到聽牌』的估計；已聽回 0。"""
    return max(0, approx_shanten_16(mj))

def _build_counts(mj: List[int]) -> List[int]:
    cnt = [0] * N_TILE_TYPES
    for v in mj:
        if 0 <= v < N_TILE_TYPES:
            cnt[v] += 1
    return cnt

def is_singleton_tile_pre_discard(tile: int, hand_before: List[int]) -> bool:
    """
    判斷出牌前該 tile 是否為孤張（±1 或 ±2 都算有靠到）：
    - 同牌只有 1 張
    - 數牌：若同花色內 ±1 或 ±2 的牌任一存在就不算孤張
    - 字牌：單張即孤張
    """
    cnt = _build_counts(hand_before)
    if not (0 <= tile < N_TILE_TYPES):
        return False
    if cnt[tile] != 1:
        return False

    # 字牌：單張就算孤張
    if tile >= 27:
        return True

    suit = tile // 9
    idx = tile % 9
    # 查 ±1, ±2 是否有牌存在
    offsets = [-2, -1, 1, 2]
    for off in offsets:
        pos = idx + off
        if 0 <= pos <= 8:
            neighbor = suit * 9 + pos
            if cnt[neighbor] > 0:
                return False  # 有靠到
    return True


def has_any_singleton(hand_before: List[int]) -> bool:
    cnt = _build_counts(hand_before)
    for t in range(N_TILE_TYPES):
        if cnt[t] == 1:
            # 字牌孤張
            if t >= 27:
                return True
            # 數牌：±1 或 ±2 有靠就不算孤張
            suit = t // 9
            idx = t % 9
            offsets = [-2, -1, 1, 2]
            has_neighbor = False
            for off in offsets:
                pos = idx + off
                if 0 <= pos <= 8:
                    neighbor = suit * 9 + pos
                    if cnt[neighbor] > 0:
                        has_neighbor = True
                        break
            if not has_neighbor:
                return True
    return False
def count_honor_singletons(hand: List[int]) -> int:
    cnt = _build_counts(hand)
    c = 0
    for t in range(27, N_TILE_TYPES):  # 27~33
        if cnt[t] == 1:
            c += 1
    return c
# ------------------------------------------------------
# 環境：Player0 = RL，其他玩家 = Heuristic
# ------------------------------------------------------
class TwMahjongEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = random.Random(seed)

        # 觀測：手牌 34 維計數（0..4）
        self.observation_space = spaces.Box(low=0, high=4, shape=(N_TILE_TYPES,), dtype=np.int8)
        # 動作：丟哪一張牌（0..33）
        self.action_space = spaces.Discrete(N_TILE_TYPES)

        # 狀態
        self.wall: List[int] = []
        self.hands: List[List[int]] = [[] for _ in range(PLAYERS)]
        self.discards: List[List[int]] = [[] for _ in range(PLAYERS)]
        self.cur_player: int = 0
        self.done: bool = False

        # 記錄 RL 獎勵輔助
        self.prev_shanten: Optional[int] = None
        self.tenpai_awarded: bool = False  # 首次進入聽牌的獎勵
        self.prev_progress: Optional[int] = None  # 這步前的「幾進聽」

    # ---------- 麻將牆與發牌 ----------
    def _build_wall(self):
        self.wall = []
        for t in range(N_TILE_TYPES):   # 0..33 每種各4張；花牌略過
            self.wall += [t, t, t, t]
        self.rng.shuffle(self.wall)

    def _draw(self) -> Optional[int]:
        if not self.wall:
            return None
        return self.wall.pop()

    def _deal(self):
        for p in range(PLAYERS):
            self.hands[p] = []
        for _ in range(START_HAND):     # 發 16 張
            for p in range(PLAYERS):
                self.hands[p].append(self._draw())
        for p in range(PLAYERS):
            self.hands[p].sort()

    # ---------- Gym 介面 ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self._build_wall()
        self._deal()
        self.cur_player = 0
        self.discards = [[] for _ in range(PLAYERS)]
        self.done = False
        self.tenpai_awarded = False

        self.prev_shanten = approx_shanten_16(self.hands[0])
        self.prev_progress = steps_to_tenpai(self.hands[0])   # ★ 全域函式
        obs = self._obs()
        return obs, {}

    def _obs(self) -> np.ndarray:
        cnt = np.zeros((N_TILE_TYPES,), dtype=np.int8)
        for v in self.hands[0]:
            if 0 <= v < N_TILE_TYPES:
                cnt[v] += 1
        return cnt

    def step(self, action: int):
        """
        1) 所有人先摸牌
        2) Player0 丟牌（RL）；其餘用啟發式
        3) 簡化終局：牆空或任一方達成「向聽==0 且張數為 3n+2」視為和牌
        """
        assert not self.done

        reward = 0.0
        info: Dict = {}

        # --- 1) 摸牌 ---
        for p in range(PLAYERS):
            card = self._draw()
            if card is not None:
                self.hands[p].append(card)
                self.hands[p].sort()

        # --- 2) 丟牌 ---
        reward += self._rl_discard_and_reward(action)

        # 其他玩家（啟發式）
        for p in range(1, PLAYERS):
            self._heuristic_move(p)

        # --- 3) 檢查終局 ---
        WIN_REWARD_SELF   = 10.0  # 自己胡
        WIN_PENALTY_OTHER = -2.0  # 他人胡
        for p in range(PLAYERS):
            sh = approx_shanten_16(self.hands[p])
            if sh == 0 and len(self.hands[p]) % 3 == 2:
                if p == 0:
                    reward += WIN_REWARD_SELF
                else:
                    reward += WIN_PENALTY_OTHER
                self.done = True
                info["winner"] = p
                break

        if not self.wall and not self.done:
            self.done = True
            info["draw"] = True

        obs = self._obs()
        terminated = self.done
        truncated = False
        return obs, reward, terminated, truncated, info

    # ---------- 動作細節 ----------
    def _rl_discard_and_reward(self, action: int) -> float:
        STEP_PENALTY               = -0.1
        BIG_REWARD_PROGRESS        =  +0.5
        REWARD_DISCARD_SINGLETON   =  +0.20
        BONUS_HONOR_SINGLETON      =  +0.15
        PENALTY_MISSED_SINGLETON   =  -0.50

        reward = STEP_PENALTY

        old_prog = self.prev_progress if self.prev_progress is not None else steps_to_tenpai(self.hands[0])

        # 1) 先處理暗槓
        dk = dark_kong_available(self.hands[0])
        if dk is not None and self.hands[0].count(dk) >= 4:
            for _ in range(4):
                self.hands[0].remove(dk)
            card = self._draw()
            if card is not None:
                self.hands[0].append(card)
                self.hands[0].sort()

        # 2) 決定要丟的牌
        target = action
        if target < 0 or target >= N_TILE_TYPES or self.hands[0].count(target) == 0:
            reward -= 0.05
            target = heuristic_discard(self.hands[0])

        hand_before = self.hands[0][:]
        had_singleton_before = has_any_singleton(hand_before)
        is_target_single = is_singleton_tile_pre_discard(target, hand_before)

        # 極端保底
        if self.hands[0].count(target) == 0:
            if len(self.hands[0]) == 0:
                self.prev_progress = old_prog
                return reward
            target = self.hands[0][0]
            hand_before = self.hands[0][:]
            had_singleton_before = has_any_singleton(hand_before)
            is_target_single = is_singleton_tile_pre_discard(target, hand_before)

        # 3) 真正出牌
        self.hands[0].remove(target)
        self.discards[0].append(target)

        # 4) 看向聽有沒有變好
        new_prog = steps_to_tenpai(self.hands[0])
        if new_prog < old_prog:
            reward += BIG_REWARD_PROGRESS
            self.prev_progress = new_prog
            return reward

        # 5) 沒變好才看孤張
        if is_target_single:
            reward += REWARD_DISCARD_SINGLETON
            if target >= 27:
                reward += BONUS_HONOR_SINGLETON
        else:
            if had_singleton_before:
                reward += PENALTY_MISSED_SINGLETON
                # 有大字孤張沒丟 → 再扣
                honor_sing_cnt_before = count_honor_singletons(hand_before)
                if honor_sing_cnt_before > 0:
                    reward += PENALTY_MISSED_SINGLETON * honor_sing_cnt_before  # 每張再 -0.20

        # 6) 持有成本：步末還拿著大字孤張 → 扣
        honor_sing_cnt_after = count_honor_singletons(self.hands[0])
        if honor_sing_cnt_after > 0:
            reward -= 0.05 * honor_sing_cnt_after

        self.prev_progress = new_prog
        return reward


    def _heuristic_move(self, p: int):
        """對手丟牌（可加上暗槓邏輯）。"""
        dk = dark_kong_available(self.hands[p])
        if dk is not None and self.hands[p].count(dk) >= 4:
            for _ in range(4):
                self.hands[p].remove(dk)
            card = self._draw()
            if card is not None:
                self.hands[p].append(card)
                self.hands[p].sort()

        v = heuristic_discard(self.hands[p])
        self.hands[p].remove(v)
        self.discards[p].append(v)

# ======================================================
# 訓練腳本（Stable-Baselines3 / PPO）多環境並行 + GPU
# ======================================================
if __name__ == "__main__":
    import os
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    # ---- 推薦：允許 TF32（3080 適用）
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ---- 多環境工廠（Windows 一定要放在 __main__ 內）
    def make_env(seed_offset=0):
        def _f():
            return TwMahjongEnv(seed=42 + seed_offset)
        return _f

    # ★ 依 CPU 執行緒數設定；先從 8 起（可調 8~12）
    N_ENVS = min(8, (os.cpu_count() or 8))
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # ---- 重要：n_steps * N_ENVS 必須能被 batch_size 整除
    N_STEPS = 8192
    BATCH   = 1024           # 8192 * 8 = 65536，可整除 1024

    device_str = "cpu"#use cpu is better
    print("Using", device_str, "device")

    model = PPO(
        "MlpPolicy",
        vec_env,
        device=device_str,
        verbose=1,
        n_steps=N_STEPS,
        batch_size=BATCH,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256, 256])
    )

    # ---- 長一點的訓練比較有感
    TOTAL_STEPS = 20_000_000
    model.learn(total_timesteps=TOTAL_STEPS)

    os.makedirs("models", exist_ok=True)
    model.save("models/tw_mahjong_ppo_gpu")
    print("Model saved to models/tw_mahjong_ppo_gpu.zip")

    # ===== 簡單測試（單環境即可） =====
    env = TwMahjongEnv(seed=123)
    obs, _ = env.reset()
    total_r = 0.0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(int(action))
        total_r += r
        if term or trunc:
            print("Episode end:", info)
            obs, _ = env.reset()
    print("Test total reward:", total_r)
