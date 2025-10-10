#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time
import traceback
from functools import cached_property
from typing import Any

try:
    import rclpy
except ImportError:
    rclpy = None

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so101_follower import SO101FollowerConfig

try:
    from .ros2_motor_executor import MotorExecutorNode
except ImportError:
    MotorExecutorNode = None

logger = logging.getLogger(__name__)


class SO101Follower(Robot):
    """
    SO-101 Follower Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101FollowerConfig
    name = "so101_follower"

    def __init__(self, config: SO101FollowerConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # 初始化ROS2和MotorExecutorNode
        self._init_ros2()

        # 创建定时器线程，每10ms执行get_action并存储到motor_executor
        self._timer_running = False
        self._timer_thread = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        # 启动定时器线程
        self._start_action_timer()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # self.calibration is not empty here
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(
                    f"Writing calibration file associated with the id {self.id} to the motors"
                )
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write(
                        "Max_Torque_Limit", motor, 500
                    )  # 50% of the max torque limit to avoid burnout
                    self.bus.write(
                        "Protection_Current", motor, 250
                    )  # 50% of max current to avoid burnout
                    self.bus.write(
                        "Overload_Torque", motor, 25
                    )  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(
                f"Connect the controller board to the '{motor}' motor only and press enter."
            )
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        print(f"pan goal_pos values: {goal_pos.values()}")
        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def get_action_cmd(self) -> dict[str, Any] | None:
        """从 ROS2 motor executor 中获取动作命令

        Returns:
            dict[str, Any] | None: 从 motor executor 队列中获取的动作命令，如果没有可用命令则返回 None
        """
        if not hasattr(self, "motor_executor") or self.motor_executor is None:
            logger.warning("Motor executor not initialized, cannot get action command")
            return None

        try:
            # 从 motor_executor 获取 follower_action
            action_str = self.motor_executor.pop_follower_action()

            if action_str is None:
                return None

            # 如果是字符串，尝试解析为 JSON
            if isinstance(action_str, str):
                try:
                    import json

                    action_dict = json.loads(action_str)
                    # print(f"Parsed action command: {action_dict}")
                    return action_dict
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse action command as JSON: {action_str}"
                    )
                    return {"raw_command": action_str}
            else:
                # 如果不是字符串，直接返回
                return action_str

        except Exception as e:
            logger.error(f"Error getting action command from motor executor: {e}")
            return None

    def _start_action_timer(self):
        """启动定时器线程，每10ms执行get_action并存储到motor_executor"""
        if self._timer_running:
            return

        self._timer_running = True
        self._timer_thread = threading.Thread(target=self._timer_worker, daemon=True)
        self._timer_thread.start()
        logger.info("Action timer started (10ms interval)")

    def _stop_action_timer(self):
        """停止定时器线程"""
        if not self._timer_running:
            return

        self._timer_running = False
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=1.0)
        logger.info("Action timer stopped")

    def _timer_worker(self):
        """定时器工作线程，每10ms执行一次"""
        while self._timer_running:
            try:
                if self.is_connected:
                    # 获取当前action
                    action = self._get_action_internal()
                    # 存储到motor_executor
                    if self.motor_executor is not None:
                        self.motor_executor.set_motor_state(action)
            except Exception as e:
                logger.error(f"Timer worker error: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")

            time.sleep(0.01)  # 10ms间隔

    def _get_action_internal(self) -> dict[str, float]:
        """内部使用的get_action方法，不包含debug日志"""
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        return action

    def _register_motor_executor_methods(self):
        """Register _normalize and _unnormalize methods to motor_executor"""
        if self.motor_executor is None:
            logger.warning(
                "Motor executor not available, cannot register normalize methods"
            )
            return

        try:
            # 获取bus的normalize和unnormalize方法
            normalize_method = self.bus._normalize
            unnormalize_method = self.bus._unnormalize

            # 使用专门的函数来注册callback
            self.motor_executor.set_bus_normalize_callback(normalize_method)
            self.motor_executor.set_bus_unnormalize_callback(unnormalize_method)
            self.motor_executor.set_get_motor_id_callback(self.bus._get_motor_id)

            logger.info(
                "Successfully registered _normalize and _unnormalize methods to motor_executor"
            )

        except Exception as e:
            logger.error(f"Error registering normalize methods to motor_executor: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")

    def _init_ros2(self):
        """Initialize ROS2 and create MotorExecutorNode"""
        try:
            # 检查是否有 rclpy 模块
            if rclpy is None:
                raise ImportError("rclpy module not available")

            # 检查是否有 MotorExecutorNode 类
            if MotorExecutorNode is None:
                raise ImportError("MotorExecutorNode not available")

            # 检查rclpy是否已经初始化
            if not rclpy.ok():
                rclpy.init()
                self._rclpy_initialized_by_us = True
            else:
                self._rclpy_initialized_by_us = False

            # 创建MotorExecutorNode
            self.motor_executor = MotorExecutorNode(arm_side=self.config.arm_side)
            logger.info("ROS2 MotorExecutorNode initialized successfully")

            # 注册 _normalize 和 _unnormalize 方法到 motor_executor
            self._register_motor_executor_methods()

        except ImportError as e:
            logger.warning(
                f"ROS2/rclpy not available: {e}. Running without ROS2 integration."
            )
            self.motor_executor = None
            self._rclpy_initialized_by_us = False
        except Exception as e:
            logger.warning(f"Failed to initialize ROS2 MotorExecutorNode: {e}")
            self.motor_executor = None
            self._rclpy_initialized_by_us = False

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 停止定时器
        self._stop_action_timer()

        # 关闭motor_executor
        if hasattr(self, "motor_executor") and self.motor_executor is not None:
            self.motor_executor.shutdown()

        # 如果我们初始化了rclpy，则需要关闭它
        if hasattr(self, "_rclpy_initialized_by_us") and self._rclpy_initialized_by_us:
            try:
                if rclpy is not None:
                    rclpy.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down rclpy: {e}")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
