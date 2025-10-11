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
        if tau <= 0.0:
            self._tau = 0.2
        if tau > 0.0:
            self._tau = float(tau)
        self._v_limit = float(v_limit) if v_limit is not None else None
        self._a_limit = float(a_limit) if a_limit is not None else None
        if vel_ema_alpha < 0.0:
            self._vel_ema_alpha = 0.0
        if vel_ema_alpha > 1.0:
            self._vel_ema_alpha = 1.0
        if 0.0 <= vel_ema_alpha <= 1.0:
            self._vel_ema_alpha = float(vel_ema_alpha)
        self._dup_eps: Optional[float] = None
        if duplicate_epsilon is not None:
            if duplicate_epsilon > 0.0:
                self._dup_eps = float(duplicate_epsilon)
        self._prev_meas: Optional[float] = None
        self._x: Optional[float] = None
        self._vel: float = 0.0

    def update(self, pos: float, target: float, dt: float) -> float:
        """二阶临界阻尼设定点平滑更新。

        参数:
            pos (float): 当前测得的位置值。单位需与 target 一致。
            target (float): 期望到达的位置设定点。
            dt (float): 本帧与上一帧之间的时间间隔(秒)。需 > 0。

        返回:
            float: 平滑后的下一帧位置 pos_next。

        说明:
            - 系统采用临界阻尼形式, ω = 2/τ, 加速度 a = ω²(target-pos) - 2ω·v。
            - 速度 v 由位置微分估计, 并可通过 vel_ema_alpha 进行 EMA 平滑。
            - 若配置了 a_limit、v_limit, 将分别对加速度与速度进行限幅。
            - 当 dt<=0 时直接返回 pos; 当 τ 很小(≤1e-6)时直接返回 target。
        """
        if dt <= 0.0:
            if self._x is not None:
                return float(self._x)
            return float(pos)
        if self._x is None:
            self._x = float(pos)
            self._prev_meas = float(pos)
            self._vel = 0.0
            return float(self._x)
        if self._tau <= 1e-6:
            self._vel = 0.0
            self._x = float(target)
            self._prev_meas = float(pos)
            return float(self._x)

        is_duplicate = False
        if self._dup_eps is not None:
            if self._prev_meas is not None:
                if abs(float(pos) - self._prev_meas) <= self._dup_eps:
                    is_duplicate = True

        if is_duplicate:
            w = 2.0 / self._tau
            a = (w * w) * (float(target) - float(self._x)) - 2.0 * w * self._vel
            if self._a_limit is not None:
                if a > self._a_limit:
                    a = self._a_limit
                if a < -self._a_limit:
                    a = -self._a_limit
            v_next = self._vel + a * dt
            if self._v_limit is not None:
                if v_next > self._v_limit:
                    v_next = self._v_limit
                if v_next < -self._v_limit:
                    v_next = -self._v_limit
            x_next = float(self._x) + v_next * dt
            self._vel = v_next
            self._x = x_next
            self._prev_meas = float(pos)
            return float(self._x)

        v_meas = 0.0
        if self._prev_meas is None:
            self._prev_meas = float(pos)
        else:
            v_meas = (float(pos) - self._prev_meas) / float(dt)
        if self._vel_ema_alpha <= 0.0:
            self._vel = v_meas
        else:
            alpha = self._vel_ema_alpha
            self._vel = (1.0 - alpha) * self._vel + alpha * v_meas
        self._prev_meas = float(pos)
        w = 2.0 / self._tau
        a = (w * w) * (float(target) - float(pos)) - 2.0 * w * self._vel
        if self._a_limit is not None:
            if a > self._a_limit:
                a = self._a_limit
            if a < -self._a_limit:
                a = -self._a_limit
        v_next = self._vel + a * dt
        if self._v_limit is not None:
            if v_next > self._v_limit:
                v_next = self._v_limit
            if v_next < -self._v_limit:
                v_next = -self._v_limit
        x_next = float(pos) + v_next * dt
        self._vel = v_next
        self._x = x_next
        return float(self._x)

    def _differentiate(self, pos: float, dt: float) -> float:
        if dt <= 0.0:
            return float(self._vel)
        if self._prev_meas is None:
            self._prev_meas = float(pos)
            self._vel = 0.0
            return float(self._vel)
        raw_v = (float(pos) - self._prev_meas) / float(dt)
        if self._vel_ema_alpha <= 0.0:
            self._vel = raw_v
            self._prev_meas = float(pos)
            return float(self._vel)
        alpha = self._vel_ema_alpha
        self._vel = (1.0 - alpha) * self._vel + alpha * raw_v
        self._prev_meas = float(pos)
        return float(self._vel)

    # Only internal helpers remain below
