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
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

from ..robot import Robot
from .config_bi_so101_follower import BiSO101FollowerConfig

logger = logging.getLogger(__name__)


class BiSO101Follower(Robot):
    """
    [Bimanual SO-101 Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio and Hugging Face
    This bimanual robot uses SO-101 follower arms for both left and right arms.
    """

    config_class = BiSO101FollowerConfig
    name = "bi_so101_follower"

    def __init__(self, config: BiSO101FollowerConfig):
        super().__init__(config)
        self.config = config

        left_id = None
        if config.left_id:
            left_id = config.left_id
        elif config.id:
            left_id = f"{config.id}_left"

        right_id = None
        if config.right_id:
            right_id = config.right_id
        elif config.id:
            right_id = f"{config.id}_right"

        left_enable_chassis = config.chassis_with_arm_id == left_id if config.chassis_with_arm_id and left_id else False
        left_enable_head = config.head_with_arm_id == left_id if config.head_with_arm_id and left_id else False

        right_enable_chassis = config.chassis_with_arm_id == right_id if config.chassis_with_arm_id and right_id else False
        right_enable_head = config.head_with_arm_id == right_id if config.head_with_arm_id and right_id else False

        self.left_enable_chassis = left_enable_chassis
        self.right_enable_chassis = right_enable_chassis

        left_arm_config = SO101FollowerConfig(
            id=left_id,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            arm_side="left",
            enable_chassis=left_enable_chassis,
            enable_head=left_enable_head,
            cameras={},
        )

        right_arm_config = SO101FollowerConfig(
            id=right_id,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            arm_side="right",
            enable_chassis=right_enable_chassis,
            enable_head=right_enable_head,
            cameras={},
        )

        self.left_arm = SO101Follower(left_arm_config)
        self.right_arm = SO101Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

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
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def get_action_cmd(self) -> dict[str, Any] | None:
        """从 ROS2 motor executor 中获取双臂动作命令

        Returns:
            dict[str, Any] | None: 合并的左右臂动作命令，如果没有可用命令则返回 None
        """
        left_action = self.left_arm.get_action_cmd()
        right_action = self.right_arm.get_action_cmd()

        if left_action is None and right_action is None:
            return None

        combined_action = {}

        if left_action is not None:
            combined_action.update({f"left_{key}": value for key, value in left_action.items()})

        if right_action is not None:
            combined_action.update({f"right_{key}": value for key, value in right_action.items()})

        return combined_action

    def get_chassis_cmd(self) -> dict[str, Any] | None:
        """从 ROS2 motor executor 中获取底盘命令

        Returns:
            dict[str, Any] | None: 从使能底盘的手臂获取的底盘命令，如果没有可用命令则返回 None
        """
        if self.left_enable_chassis:
            left_chassis = self.left_arm.get_chassis_cmd()
            if left_chassis is not None:
                return {f"left_{key}": value for key, value in left_chassis.items()}

        if self.right_enable_chassis:
            right_chassis = self.right_arm.get_chassis_cmd()
            if right_chassis is not None:
                return {f"right_{key}": value for key, value in right_chassis.items()}

        return None

    def send_action(self, action: dict[str, Any] | None = None) -> dict[str, Any]:
        """发送动作到双臂

        Args:
            action: 可选的动作字典。如果为 None，则自动从 motor executor 获取动作命令并发送

        Returns:
            dict[str, Any]: 实际发送的动作
        """
        if action is None:
            action = self.get_action_cmd()
            if action is None:
                return {}

        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = {}
        send_action_right = {}

        if len(left_action) != 0:
            send_action_left = self.left_arm.send_action(left_action)
            print(f"Sending left arm action: {left_action}")
        if len(right_action) != 0:
            print(f"Sending right arm action: {right_action}")
            send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
