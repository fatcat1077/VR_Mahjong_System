# mj_utils.py
# 共用小工具：避免 p16mj 與 hu_result 互相 import 造成循環

def next_not_block(block, mj_num, next=0):
    """
    回傳從 next 起第一個 block[i] == 0 的索引；找不到回傳 -1
    """
    i = next
    while i < mj_num:
        if block[i] == 0:
            return i
        i += 1
    return -1

def next_two_not_block(block, mj_num, next):
    """
    連續找兩個未被 block 的索引；若其中任一找不到以 -1 表示
    """
    n0 = next_not_block(block, mj_num, next)
    if n0 == -1:
        return -1, -1
    n1 = next_not_block(block, mj_num, n0 + 1)
    if n1 == -1:
        return n0, -1
    return n0, n1

def next_not_blsame(block, mj_num, mj, sv, next=0):
    """
    從 next 起找第一個 block[i] == 0 且 mj[i] != sv 的索引；找不到回傳 -1
    """
    i = next
    while i < mj_num:
        if block[i] == 0 and mj[i] != sv:
            return i
        i += 1
    return -1

def next_two_not_blsame(block, mj_num, next, mj, sv):
    """
    找兩個符合「未被 block 且牌值 != sv」的索引；若其中任一找不到以 -1 表示
    """
    n0 = next_not_blsame(block, mj_num, mj, sv, next)
    if n0 == -1:
        return -1, -1
    n1 = next_not_blsame(block, mj_num, mj, mj[n0], n0 + 1)
    if n1 == -1:
        return n0, -1
    return n0, n1
