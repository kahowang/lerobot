#!/usr/bin/env python3

from typing import Optional


class CriticallyDampedSmoother1D:
    def __init__(
        self,
        tau: float = 0.2,
        v_limit: Optional[float] = None,
        a_limit: Optional[float] = None,
        vel_ema_alpha: float = 0.0,
        duplicate_epsilon: Optional[float] = None,
    ) -> None:
        self._tau = float(tau) if tau > 0.0 else 0.2
        self._v_limit = float(v_limit) if v_limit is not None else None
        self._a_limit = float(a_limit) if a_limit is not None else None
        if vel_ema_alpha < 0.0:
            self._vel_ema_alpha = 0.0
        elif vel_ema_alpha > 1.0:
            self._vel_ema_alpha = 1.0
        else:
            self._vel_ema_alpha = float(vel_ema_alpha)
        self._dup_eps: Optional[float] = (
            float(duplicate_epsilon)
            if (duplicate_epsilon is not None and duplicate_epsilon > 0.0)
            else None
        )
        self._vel: float = 0.0

    def update(self, pos: float, target: float, dt: float) -> float:
        """速度平滑+限幅, 不进行预测。

        参数:
            pos (float): 当前测得的位置。
            target (float): 期望的位置设定点。
            dt (float): 与上一帧的时间间隔(秒), 需>0。

        返回:
            float: 经过速度平滑与限幅后的下一帧位置。
        """
        if dt <= 0.0:
            return float(pos)
        err = float(target) - float(pos)
        v_cmd = err / float(dt)
        if self._vel_ema_alpha <= 0.0:
            v = v_cmd
        else:
            alpha = self._vel_ema_alpha
            v = (1.0 - alpha) * self._vel + alpha * v_cmd
        if self._v_limit is not None:
            if v > self._v_limit:
                v = self._v_limit
            if v < -self._v_limit:
                v = -self._v_limit
        x_next = float(pos) + v * float(dt)
        self._vel = float(v)
        return float(x_next)

    def _differentiate(self, pos: float, dt: float) -> float:
        if dt <= 0.0:
            return float(self._vel)
        raw_v = 0.0
        # 仅作为便捷估计器: 使用上一帧速度与当前微分做EMA
        # 这里不缓存上一次pos, 调用方若需可自行在外部维护
        # 原类接口保留以兼容, 仅返回平滑速度
        if self._vel_ema_alpha <= 0.0:
            raw_v = 0.0
            return float(self._vel)
        alpha = self._vel_ema_alpha
        self._vel = (1.0 - alpha) * self._vel + alpha * raw_v
        return float(self._vel)

    # Only internal helpers remain below
