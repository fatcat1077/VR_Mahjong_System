# play_with_pygame_env.py
# 載入訓練好的 SB3 PPO 模型，將 Player0 改為 RL 出牌，其它家維持原本啟發式 AI。
import numpy as np
from stable_baselines3 import PPO

# 這裡請換成你的原始環境檔名（就是那支 Pygame 的大檔）
import p16mj as game

MODEL_PATH = "models/tw_mahjong_ppo_gpu.zip"

# === 觀測：把 Player0 手牌做成 34 維計數向量（與訓練時一致） ===
def hand_to_obs_34count(hand):
    cnt = np.zeros((34,), dtype=np.int8)
    for v in hand:
        if 0 <= v < 34:
            cnt[v] += 1
    return cnt

# === 啟發式丟牌（用原程式提供的工具函式）===
def heuristic_discard_from_game_module(hand):
    tmj = sorted(hand[:])
    block = [0] * len(tmj)
    block = game.add_block3(tmj, len(tmj), block)
    if block.count(0) > 2:
        block = game.add_block2(tmj, len(tmj), block)
    # 找第一張沒被封塊的
    di = game.next_not_block(block, len(block))
    if di == -1:
        return tmj[-1]
    return tmj[di]

# === 用 RL 決策的 mjAI：覆寫原本的 mjAI（只針對 Player0）===
def rl_mjAI(tid, getv=None):
    """
    盡量遵守原 mjAI 的介面與副作用：
      - 丟出一張牌後：更新 game.drop_mj, game.player_mj, game.player_mj_num
      - 重畫畫面、延遲與回傳值：丟牌完成回傳 2；若做了加槓/補花則回傳 -1（沿用原約定）
    """
    # 僅 Player0 使用 RL；其他玩家繼續用舊邏輯
    if tid != 0:
        return game.mjAI_original(tid, getv)

    # 1) 起手/摸到花 -> 補花（沿用原函式）
    if getv is not None:
        if game.proc_add_hmj(tid, True, getv) == 1:
            # 有補花，原本流程會再摸牌，這回合先返回 -1
            return -1

    # 2) 加槓（沿用原邏輯）
    if getv is not None:
        add_kong_info = game.player_add_kong(game.dmj[tid], game.player_mj[tid], getv)
        if add_kong_info is not None:
            aki, vi, game.player_mj[tid], game.player_mj_num[tid] = add_kong_info
            game.screen.blit(game.write(u"加槓", (0, 0, 255)), game.htext_loc[tid])
            game.dmj[tid][aki] = [1, [getv]]
            del game.player_mj[tid][vi]
            game.player_mj_num[tid] = len(game.player_mj[tid])
            game.pygame.display.update()
            if game.Add_Delay: game.delay(1*game.step)
            game.add_kong_mj = aki
            return -1  # 與原 mjAI 一致：做了加槓這回合先結束

    # 3) 把 getv 插入手牌（如同原流程）
    if getv is not None:
        tmj, tmj_num = game.hu_result.insert_mj(getv, game.player_mj[tid])
        # 只有玩家0需要設 getmj 顯示（原本有用到）
        if tid == 0:
            game.getmj = getv
    else:
        tmj = game.player_mj[tid][:]
        tmj_num = len(tmj)

    # 4) 先處理「暗槓」：避免與丟牌衝突
    #  （這裡沿用簡單策略：如果有四張就暗槓；你也可以關掉，以免太複雜）
    #   注意：原程式暗槓後會畫面提示與補摸；我們也維持一致。
    #   但為了穩定，暗槓後本回合回傳 -1（交給主流程處理後續）。
    value_counts = {}
    for v in tmj:
        value_counts[v] = value_counts.get(v, 0) + 1
    four_kind = [v for v, c in value_counts.items() if c >= 4]
    if four_kind:
        dk = four_kind[0]
        # 從手牌移除四張
        removed = 0
        new_hand = []
        for v in tmj:
            if v == dk and removed < 4:
                removed += 1
            else:
                new_hand.append(v)
        game.player_mj[tid] = new_hand
        game.player_mj_num[tid] = len(new_hand)
        game.dmj[tid].append([2, [dk]])  # 暗槓
        game.screen.blit(game.write(u"暗槓", (0, 0, 255)), game.htext_loc[tid])
        game.pygame.display.update()
        if game.Add_Delay: game.delay(1*game.step)
        return -1

    # 5) 以 RL 模型選擇丟哪張
    obs = hand_to_obs_34count(tmj)
    action, _ = rl_model.predict(obs, deterministic=True)
    target = int(action)

    # 無效動作：手上沒有該牌 -> 改用啟發式
    if tmj.count(target) == 0:
        target = heuristic_discard_from_game_module(tmj)

    # 6) 更新手牌、河、畫面（同原 mjAI）
    game.drop_mj[tid].append(target)
    # 從 tmj 移除 target 後，回填到原全域結構
    removed_once = False
    new_hand = []
    for v in tmj:
        if not removed_once and v == target:
            removed_once = True
            continue
        new_hand.append(v)
    game.player_mj[tid] = new_hand
    game.player_mj_num[tid] = len(new_hand)

    game.display_all(game.winner)
    game.pygame.display.update()
    if game.Add_Delay: game.delay(1*game.step)

    return 2  # 丟牌完成

if __name__ == "__main__":
    # 載入模型
    rl_model = PPO.load(MODEL_PATH, device="cpu")

    # 把原本的 mjAI 暫存，讓非 Player0 時可以沿用
    game.mjAI_original = game.mjAI
    # 覆寫成 RL 版（Player0 用 RL，其他家呼叫原版）
    game.mjAI = rl_mjAI

    # 讓 Player0 走 AI 路徑（原程式遇到 p0_is_AI==True 才會走 AI 分支）
    game.p0_is_AI = True

    # 開始原本的 Pygame 遊戲回圈
    game.main()
