from dataclasses import dataclass, field
from math import pi


@dataclass
class ChassisConvertMotor:
    wheel_distance: float
    wheel_diameter: float
    linear_rate: float = 1.0
    angular_rate: float = 1.0
    motor_max_velocity: int = 3000
    motor_indices: int = field(default=4096, init=False)

    def __post_init__(self):
        if self.wheel_distance <= 0:
            raise ValueError("wheel_distance must be positive")
        if self.wheel_diameter <= 0:
            raise ValueError("wheel_diameter must be positive")
        if self.motor_indices <= 0:
            raise ValueError("motor_indices must be positive")

        self.circumference = pi * self.wheel_diameter
        if self.circumference <= 0:
            raise ValueError("wheel circumference must be positive")

        self.counts_per_meter = self.motor_indices / self.circumference

    def convert(self, linear_x: float, angular_z: float) -> tuple[float, float]:
        scaled_linear_x = linear_x * self.linear_rate
        scaled_angular_z = angular_z * self.angular_rate

        half_wheel_distance = self.wheel_distance / 2.0
        angular_contribution = scaled_angular_z * half_wheel_distance

        left_wheel_velocity = scaled_linear_x - angular_contribution
        right_wheel_velocity = scaled_linear_x + angular_contribution

        return left_wheel_velocity, right_wheel_velocity

    def wheel_speed_to_counts(self, wheel_linear_speed: float, duration: float = 1.0) -> int:
        if duration <= 0:
            raise ValueError("duration must be positive")
        counts = wheel_linear_speed * self.counts_per_meter * duration
        max_counts = int(round(self.motor_max_velocity * duration))
        if max_counts <= 0:
            return 0
        if counts > max_counts:
            return max_counts
        if counts < -max_counts:
            return -max_counts
        return int(round(counts))

    def convert_to_motor_counts(
        self, linear_x: float, angular_z: float, duration: float = 1.0
    ) -> tuple[int, int]:
        left_wheel_velocity, right_wheel_velocity = self.convert(linear_x, angular_z)
        left_counts = self.wheel_speed_to_counts(left_wheel_velocity, duration)
        right_counts = self.wheel_speed_to_counts(right_wheel_velocity, duration)
        return left_counts, right_counts
