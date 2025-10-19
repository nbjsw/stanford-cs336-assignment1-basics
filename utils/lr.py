import math

def calculate_cosine_annealing_lr(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    """
    计算给定迭代步数 t 时的学习率 alpha_t。

    参数:
        t (int): 当前迭代步数。
        alpha_max (float): 最大学习率。
        alpha_min (float): 最小(最终)学习率。
        Tw (int): 热身迭代步数。
        Tc (int): 余弦退火迭代步数 (通常是总训练步数)。
    """
    # 热身阶段 (Warm-up): If t < Tw
    if t < Tw:
        # 公式: alpha_t = (t / Tw) * alpha_max
        # 确保 Tw 不为零以防除零错误
        if Tw == 0:
            return alpha_max
        return (t / Tw) * alpha_max

    # 余弦退火阶段 (Cosine annealing): If Tw <= t <= Tc
    elif t <= Tc:
        # 公式: alpha_t = alpha_min + 0.5 * (1 + cos(((t - Tw) / (Tc - Tw)) * pi)) * (alpha_max - alpha_min)

        # 确保分母 (Tc - Tw) 不为零
        if Tc == Tw:
            return alpha_min

        # 1. 归一化步数 (Normalized time in [0, 1])
        normalized_t = (t - Tw) / (Tc - Tw)

        # 2. 余弦衰减项 (Cosine decay factor)
        cosine_decay = 0.5 * (1 + math.cos(normalized_t * math.pi))

        # 3. 计算最终学习率
        return alpha_min + cosine_decay * (alpha_max - alpha_min)

    # 退火后阶段 (Post-annealing): If t > Tc
    else:
        # 公式: alpha_t = alpha_min
        return alpha_min
