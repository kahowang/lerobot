from dataclasses import dataclass, field
from math import pi, sqrt
from typing import Tuple


@dataclass
class ThreeWheelsChassisConvertMotor:
	wheel_distance: float
	wheel_diameter: float
	linear_rate: float = 1.0
	angular_rate: float = 1.0
	motor_max_velocity: int = 30000
	motor_indices: int = field(default=4096, init=False)
	motor1_clockwise: int = 1  # wheel at 0°
	motor2_clockwise: int = 1  # wheel at 120°
	motor3_clockwise: int = 1  # wheel at 240°

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
		self.L = self.wheel_distance / 2.0

		if self.motor1_clockwise == 0 or self.motor2_clockwise == 0 or self.motor3_clockwise == 0:
			raise ValueError("motor*_clockwise must be non-zero (use 1 or -1)")

		self._sin120 = sqrt(3.0) / 2.0
		self._sin240 = -self._sin120

	def convert(self, linear_x: float, angular_z: float) -> Tuple[float, float, float]:
		scaled_linear_x = linear_x * self.linear_rate
		scaled_angular_z = angular_z * self.angular_rate

		w1 = self.motor1_clockwise * (0.0 * scaled_linear_x + self.L * scaled_angular_z)
		w2 = self.motor2_clockwise * (-self._sin120 * scaled_linear_x + self.L * scaled_angular_z)
		w3 = self.motor3_clockwise * (-self._sin240 * scaled_linear_x + self.L * scaled_angular_z)

		return w1, w2, w3

	def wheel_speed_to_counts(self, wheel_linear_speed: float, duration: float = 1.0) -> int:
		if duration <= 0:
			raise ValueError("duration must be positive")
		counts = wheel_linear_speed * self.counts_per_meter
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
	) -> Tuple[int, int, int]:
		w1, w2, w3 = self.convert(linear_x, angular_z)
		c1 = self.wheel_speed_to_counts(w1, duration)
		c2 = self.wheel_speed_to_counts(w2, duration)
		c3 = self.wheel_speed_to_counts(w3, duration)
		return c1, c2, c3

