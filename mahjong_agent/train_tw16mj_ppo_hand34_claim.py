# train_tw16mj_ppo_hand34_claim.py
# SB3 PPO 訓練：加入吃/碰（Chi/Pon）決策 + 河牌資訊（含自己/他人打出過的牌）
#
# - env: tw16mj_gym_env_hand34_claim.Tw16MahjongEnv
# - observation_dim: 175
# - action_dim: 39 (0..33 discard, 34 pass, 35 pon, 36..38 chi variants)

import os
import multiprocessing as mp

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from tw16mj_gym_env_hand34_claim import Tw16MahjongEnv


def make_env(rank: int, seed: int = 42, max_steps: int = 500):
    def _init():
        return Tw16MahjongEnv(seed=seed + rank, max_steps=max_steps)
    return _init


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    n_envs = min(8, os.cpu_count() or 8)
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    n_steps = 4096
    batch_size = 1024

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device, "| n_envs =", n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        device=device,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    total_timesteps = 10_000_000
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    model.save("models/tw16mj_ppo_hand34_claim")
    print("Saved to models/tw16mj_ppo_hand34_claim.zip")
