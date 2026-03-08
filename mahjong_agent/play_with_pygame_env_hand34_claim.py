# play_with_pygame_env_hand34_claim.py
# - 載入 SB3 PPO zip（models/tw16mj_ppo_hand34_claim.zip）
# - 用 tw16mj_gym_env_hand34_claim.Tw16MahjongEnv 推進規則（含吃/碰）
# - 用 p16mj.py 的 Pygame GUI 顯示牌局 + 顯示模型預測（Top-K）

import time
import numpy as np
import torch
from stable_baselines3 import PPO

import p16mj as game
from tw16mj_gym_env_hand34_claim import Tw16MahjongEnv, PHASE_DISCARD, PHASE_CLAIM, ACT_PASS, ACT_PON, ACT_CHI_LOW, ACT_CHI_MID, ACT_CHI_HIGH

MODEL_PATH = "models/tw16mj_ppo_hand34_claim.zip"


def tile_to_str(t: int) -> str:
    if 0 <= t <= 8:
        return f"T{t+1}"
    if 9 <= t <= 17:
        return f"S{t-9+1}"
    if 18 <= t <= 26:
        return f"W{t-18+1}"
    if t == 27: return "東"
    if t == 28: return "南"
    if t == 29: return "西"
    if t == 30: return "北"
    if t == 31: return "白"
    if t == 32: return "發"
    if t == 33: return "中"
    if 34 <= t <= 41:
        return f"花{t-33}"
    return str(t)


def action_to_str(a: int) -> str:
    if 0 <= a <= 33:
        return f"丟 {tile_to_str(a)}"
    if a == ACT_PASS:
        return "PASS"
    if a == ACT_PON:
        return "PON(碰)"
    if a == ACT_CHI_LOW:
        return "CHI_LOW(吃)"
    if a == ACT_CHI_MID:
        return "CHI_MID(吃)"
    if a == ACT_CHI_HIGH:
        return "CHI_HIGH(吃)"
    return f"UNK({a})"


def _ensure_renderer_layout_initialized():
    if getattr(_ensure_renderer_layout_initialized, "_done", False):
        return
    _ensure_renderer_layout_initialized._done = True
    game.Add_Delay = False

    # p0_mjloc
    p0_mjloc_ini = []
    startx, starty = game.mjloc[0]
    for x in range(startx, startx + game.p_num * game.p0_mj_width, game.p0_mj_width):
        p0_mjloc_ini.append([x, starty])
    game.p0_mjloc = [list(xy) for xy in p0_mjloc_ini]
    game.p0_get_loc = list(game.p0_get_loc_org)

    # dmj/hmj/drop loc initialization（照 p16mj.main()）
    game.add_kong_loc = [[], [], [], []]
    for i, loc in enumerate(game.dmj_loc):
        (x, y) = loc[0]
        if i == 0 or i == 2:
            game.add_kong_loc[i].append((x + game.p0_mj_width, y))
        else:
            game.add_kong_loc[i].append((x, y + game.p0_mj_width))

    gap = 5
    for pi in range(4):
        for i in range(1, 5):
            (x, y) = game.dmj_loc[pi][i - 1]
            if pi == 0 or pi == 2:
                game.dmj_loc[pi].append((x + 2 * game.p0_mj_width + game.mjbk.get_width() + gap, y))
                (ax, ay) = game.dmj_loc[pi][-1]
                game.add_kong_loc[pi].append((ax + game.p0_mj_width, ay))
            else:
                game.dmj_loc[pi].append((x, y + 2 * game.p0_mj_width + game.mjbk.get_width() + gap))
                (ax, ay) = game.dmj_loc[pi][-1]
                game.add_kong_loc[pi].append((ax, ay + game.p0_mj_width))

    for pi in range(4):
        for i in range(1, 8):
            (x, y) = game.hmj_loc[pi][i - 1]
            if pi == 0 or pi == 2:
                game.hmj_loc[pi].append((x + game.p0_mj_width, y))
            else:
                game.hmj_loc[pi].append((x, y + game.p0_mj_width))
    game.hmj_loc[2] = game.hmj_loc[2][::-1]
    game.hmj_loc[3] = game.hmj_loc[3][::-1]

    for pi in range(4):
        for i in range(4):
            for j in range(8):
                if j == 0:
                    if i == 0:
                        continue
                    (x, y) = game.drop_mj_loc[pi][0]
                    if pi == 0:
                        game.drop_mj_loc[pi][i * 8] = (x, y - i * 55)
                    elif pi == 1:
                        game.drop_mj_loc[pi][i * 8] = (x - i * 55, y)
                    elif pi == 2:
                        game.drop_mj_loc[pi][i * 8] = (x, y + i * 55)
                    elif pi == 3:
                        game.drop_mj_loc[pi][i * 8] = (x + i * 55, y)
                else:
                    (x, y) = game.drop_mj_loc[pi][i * 8 + j - 1]
                    if pi == 0 or pi == 2:
                        game.drop_mj_loc[pi][i * 8 + j] = (x + game.p0_mj_width, y)
                    else:
                        game.drop_mj_loc[pi][i * 8 + j] = (x, y + game.p0_mj_width)

    for pi in range(4):
        if pi == 1 or pi == 2:
            game.drop_mj_loc[pi][0:8] = game.drop_mj_loc[pi][7::-1]
            for i in range(1, 4):
                game.drop_mj_loc[pi][8 * i:8 * (i + 1)] = game.drop_mj_loc[pi][8 * (i + 1) - 1:8 * i - 1:-1]
        game.drop_mj_loc[pi][32:64] = game.drop_mj_loc[pi][0:32]

    game.get_done = [2] * 4
    game.getmj = None
    game.calc_tai = 0
    game.winner = -1


def convert_env_melds_to_p16mj(env):
    dmj = [[] for _ in range(4)]
    for m in env.melds[0]:
        if m.kind == "PON":
            dmj[0].append([3, [m.taken]])
        elif m.kind == "CHI":
            a, b, _ = m.tiles
            dmj[0].append([0, [a, b, m.taken]])
    return dmj


def sync_renderer_from_env(env: Tw16MahjongEnv, winner: int = -1):
    game.player_mj = [env.hands[i][:] for i in range(4)]
    game.player_mj_num = [len(env.hands[i]) for i in range(4)]
    game.drop_mj = [env.discards[i][:] for i in range(4)]
    game.hmj = [env.flowers[i][:] for i in range(4)]
    game.dmj = convert_env_melds_to_p16mj(env)

    game.mjp = 0
    game.mjb = max(0, len(env.wall) - 1)
    game.turn_id = 0
    game.winner = winner


def draw_prediction_overlay(obs: np.ndarray, action: int, phase: int, topk=6):
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = rl_model.policy.get_distribution(obs_t)
        probs = dist.distribution.probs.squeeze(0).cpu().numpy()

    idx = np.argsort(-probs)[:topk]
    phase_str = "DISCARD" if phase == PHASE_DISCARD else "CLAIM"
    lines = [f"[RL] phase={phase_str} pred={action_to_str(action)}"]
    for k, a in enumerate(idx, start=1):
        lines.append(f"  {k}. {action_to_str(int(a))} : {probs[a]:.3f}")

    x, y = 20, 60
    for i, s in enumerate(lines):
        game.screen.blit(game.write(s, (255, 255, 0), 22), (x, y + i * 24))


def pump_quit_events():
    for event in game.pygame.event.get():
        if event.type == game.QUIT:
            raise SystemExit()


if __name__ == "__main__":
    _ensure_renderer_layout_initialized()

    rl_model = PPO.load(MODEL_PATH, device="cpu")

    env = Tw16MahjongEnv(seed=123, max_steps=500)
    obs, _ = env.reset()

    winner = -1
    while True:
        pump_quit_events()

        sync_renderer_from_env(env, winner=winner)
        game.display_all(winner)

        action, _ = rl_model.predict(obs, deterministic=True)
        action = int(action)

        draw_prediction_overlay(obs, action, env.phase, topk=6)
        game.pygame.display.update()

        time.sleep(0.5)

        obs, r, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            winner = info.get("winner", -1)
            sync_renderer_from_env(env, winner=winner)
            game.display_all(winner)
            game.screen.blit(game.write(f"Episode end: {info} | reward={r:.3f}", (255, 255, 255), 26), (20, 20))
            game.pygame.display.update()

            time.sleep(2.0)
            winner = -1
            obs, _ = env.reset()
