import json
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Twist

from .chassis_convert_motor import ChassisConvertMotor


class MotorExecutorNode:
    def __init__(self, arm_side="right", enable_chassis=False):
        # 创建ROS2节点作为成员变量
        self.node = Node(f"{arm_side}_motor_executor_node")

        # 创建executor
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        # 存储arm_side配置
        self.arm_side = arm_side

        # 存储enable_chassis配置
        self.enable_chassis = enable_chassis

        # 根据arm_side构建话题前缀
        topic_prefix = f"/{arm_side}" if arm_side else ""
        self.topic_prefix = topic_prefix
        # 创建发布器，发布motor_state话题
        motor_state_topic = f"{topic_prefix}/robot_control/motor_state"
        self.motor_state_publisher = self.node.create_publisher(String, motor_state_topic, 10)

        # 创建订阅器，订阅motor_cmd话题
        motor_cmd_topic = f"{topic_prefix}/robot_control/motor_cmd"
        self.motor_cmd_subscriber = self.node.create_subscription(
            String, motor_cmd_topic, self.motor_cmd_callback, 10
        )

        # 根据enable_chassis判断是否创建cmd_vel订阅器和ChassisConvertMotor
        if self.enable_chassis:
            self.cmd_vel_subscriber = self.node.create_subscription(
                Twist, "/turtle1/cmd_vel", self.cmd_vel_callback, 10
            )
            # 初始化ChassisConvertMotor，参数需要根据实际机器人配置
            self.chassis_converter = ChassisConvertMotor(
                wheel_distance=0.5,  # 轮距，单位：米
                wheel_diameter=0.1,  # 轮径，单位：米
                linear_rate=1.0,
                angular_rate=1.0,
            )
            self.node.get_logger().info("Created subscriber for /turtle1/cmd_vel")
        else:
            self.cmd_vel_subscriber = None
            self.chassis_converter = None

        # 创建定时器，定期发布motor_state
        self.timer = self.node.create_timer(0.01, self.publish_motor_state)  # 10Hz

        # 初始化motor状态
        self.motor_state = "idle"

        # 创建follower_action队列
        self.follower_action = queue.Queue()

        # 创建follower_action的递归锁
        self.follower_action_lock = threading.RLock()

        # 存储上一次的follower_action值
        self.last_follower_action = None

        # 创建chassis_action队列
        self.chassis_action = queue.Queue()

        # 创建chassis_action的递归锁
        self.chassis_action_lock = threading.RLock()

        # 存储上一次的chassis_action值
        self.last_chassis_action = None

        # 创建线程控制标志
        self.running = True

        # 创建并启动spin线程
        self.spin_thread = threading.Thread(target=self._spin_thread, daemon=True)
        self.spin_thread.start()

        self.node.get_logger().info(f"Motor Executor Node initialized with arm_side='{self.arm_side}'")
        self.node.get_logger().info(f"Publishing to: {motor_state_topic}")
        self.node.get_logger().info(f"Subscribing to: {motor_cmd_topic}")

    def execute(self, timeout_sec=0.1):
        """执行一次ROS2 spin"""
        try:
            self.executor.spin_once(timeout_sec=timeout_sec)
        except Exception as e:
            self.node.get_logger().error(f"Error in execute: {e}")

    def _spin_thread(self):
        """ROS2节点的spin线程"""
        try:
            while self.running and rclpy.ok():
                self.execute(timeout_sec=0.1)
        except Exception as e:
            self.node.get_logger().error(f"Error in spin thread: {e}")

    def motor_cmd_callback(self, msg):
        """处理接收到的motor命令"""
        action = self._cmd_string_convert_action(msg.data)
        print(f"Received {self.topic_prefix} motor command: {action}")
        # 使用递归锁保护follower_action队列的写操作
        with self.follower_action_lock:
            self.follower_action.put(action)

    def cmd_vel_callback(self, msg):
        """处理接收到的cmd_vel命令"""
        if not self.enable_chassis:
            return

        if self.chassis_converter is None:
            self.node.get_logger().warning("ChassisConverter is not initialized")
            return

        try:
            linear_x = msg.linear.x
            angular_z = msg.angular.z

            # 转换为电机counts，假设duration为0.1秒
            left_counts, right_counts = self.chassis_converter.convert_to_motor_counts(
                linear_x, angular_z, duration=0.1
            )

            # 构建chassis_action字典
            chassis_action_dict = {"chassis_left": left_counts, "chassis_right": right_counts}

            # 使用递归锁保护chassis_action队列的写操作
            with self.chassis_action_lock:
                self.chassis_action.put(chassis_action_dict)

            self.node.get_logger().debug(
                f"Received cmd_vel: linear.x={linear_x}, angular.z={angular_z}, "
                f"converted to left={left_counts}, right={right_counts}"
            )
        except Exception as e:
            self.node.get_logger().error(f"Error in cmd_vel_callback: {e}")

    def publish_motor_state(self):
        """定期发布motor状态"""
        msg = String()
        msg.data = self.motor_state
        self.motor_state_publisher.publish(msg)
        self.node.get_logger().debug(f"Published motor state: {self.motor_state}")

    def pop_follower_action(self):
        """从follower_action队列中弹出一个动作

        Returns:
            str or None: 如果队列不为空，返回队列中的第一个动作；如果队列为空，返回上一次的值
        """
        # 使用递归锁保护follower_action队列和last_follower_action的读写操作
        with self.follower_action_lock:
            try:
                action = self.follower_action.get_nowait()
                self.last_follower_action = action  # 更新上一次的值
                return action
            except queue.Empty:
                return self.last_follower_action  # 返回上一次的值

    def pop_chassis_action(self):
        """从chassis_action队列中弹出一个动作

        Returns:
            dict or None: 如果队列不为空，返回队列中的第一个动作；如果队列为空，返回上一次的值
                格式: {"chassis_left": value, "chassis_right": value}
        """
        # 使用递归锁保护chassis_action队列和last_chassis_action的读写操作
        with self.chassis_action_lock:
            try:
                action = self.chassis_action.get_nowait()
                self.last_chassis_action = action  # 更新上一次的值
                return action
            except queue.Empty:
                return self.last_chassis_action  # 返回上一次的值

    def set_motor_state(self, state_map):
        """设置motor状态

        Args:
            state_map (dict): 状态映射字典
        """
        unnormalized_map = self._unnormalize_motor_action(state_map)
        try:
            # 将字典转换为JSON字符串
            json_state = json.dumps(unnormalized_map, ensure_ascii=True, indent=None)
            self.motor_state = json_state
        except Exception as e:
            self.node.get_logger().error(f"Error setting motor state: {e}")

    def shutdown(self):
        """清理资源"""
        self.node.get_logger().info("Shutting down Motor Executor Node")

        # 停止spin线程
        self.running = False
        if hasattr(self, "spin_thread") and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
        # 清理executor
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def set_bus_normalize_callback(self, callback):
        """设置bus归一化回调函数"""
        self.bus_normalize_callback = callback

    def set_bus_unnormalize_callback(self, callback):
        """设置bus反归一化回调函数"""
        self.bus_unnormalize_callback = callback

    def set_get_motor_id_callback(self, callback):
        """设置获取电机ID回调函数"""
        self.get_motor_id_callback = callback

    def _unnormalize_motor_action(self, normalize_action):
        """将归一化的电机动作转换为反归一化的电机动作

        Args:
            normalize_action (dict): 归一化的电机动作，格式如
                {
                    "shoulder_pan.pos": 2043,
                    "shoulder_lift.pos": 1994,
                    "elbow_flex.pos": 2043,
                    "wrist_flex.pos": 2126,
                    "wrist_roll.pos": 2047,
                    "gripper.pos": 2046
                }

        Returns:
            dict: 反归一化后的电机动作映射，与输入格式相同
        """
        try:
            # 创建motor_id到motor_name的映射和motor_id到pose的映射
            id_pose_map = {}  # {motor_id: pose_value}
            name_to_id_map = {}  # {motor_name: motor_id}

            for motor_name_with_pos, pose_value in normalize_action.items():
                # 提取电机名称（去除.pos后缀）
                if motor_name_with_pos.endswith(".pos"):
                    motor_name = motor_name_with_pos[:-4]  # 移除最后4个字符 ".pos"
                else:
                    motor_name = motor_name_with_pos

                # 获取电机ID
                motor_id = self.get_motor_id_callback(motor_name)
                if motor_id is not None:
                    id_pose_map[motor_id] = pose_value
                    name_to_id_map[motor_name] = motor_id
                    self.node.get_logger().debug(
                        f"Motor '{motor_name}' -> ID {motor_id}, pose: {pose_value}"
                    )
                else:
                    self.node.get_logger().warning(
                        f"Could not get ID for motor '{motor_name}'"
                    )

            if not id_pose_map:
                self.node.get_logger().warning("No valid motor IDs found")
                return normalize_action

            # 调用bus_unnormalize_callback处理ID-pose映射
            unnormalized_map = self.bus_unnormalize_callback(id_pose_map)

            # 将unnormalized_map的结果转换回电机名称格式
            unnormalize_action = {}
            for motor_id, unnormalized_value in unnormalized_map.items():
                # 查找对应的电机名称
                motor_name = None
                for name, id_val in name_to_id_map.items():
                    if id_val == motor_id:
                        motor_name = name
                        break

                if motor_name:
                    # 生成电机键名
                    motor_key = f"{motor_name}.pos"
                    unnormalize_action[motor_key] = unnormalized_value
                    self.node.get_logger().debug(
                        f"Unnormalized {motor_key}: {unnormalized_value}"
                    )
                else:
                    self.node.get_logger().warning(
                        f"Could not find motor name for ID {motor_id}"
                    )

            return unnormalize_action

        except Exception as e:
            self.node.get_logger().error(f"Error in _unnormalize_motor_action: {e}")
            return normalize_action

    def _cmd_string_convert_action(self, json_str):
        """将JSON字符串转换成电机动作

        Args:
            json_str (str): JSON字符串，格式如 {'shoulder_pan.pos': 2042, ...}

        Returns:
            dict: 转换后的电机动作映射
        """
        try:
            # 解析JSON字符串
            origin_map = json.loads(json_str)

            # 创建motor_id到motor_name的映射和motor_id到pose的映射
            id_pose_map = {}  # {motor_id: pose_value}
            name_to_id_map = {}  # {motor_name: motor_id}

            for motor_name_with_pos, pose_value in origin_map.items():
                # 提取电机名称（去除.pos后缀）
                if motor_name_with_pos.endswith(".pos"):
                    motor_name = motor_name_with_pos[:-4]  # 移除最后4个字符 ".pos"
                else:
                    motor_name = motor_name_with_pos

                # 获取电机ID
                motor_id = self.get_motor_id_callback(motor_name)
                if motor_id is not None:
                    id_pose_map[motor_id] = pose_value
                    name_to_id_map[motor_name] = motor_id
                    self.node.get_logger().debug(
                        f"Motor '{motor_name}' -> ID {motor_id}, pose: {pose_value}"
                    )
                else:
                    self.node.get_logger().warning(
                        f"Could not get ID for motor '{motor_name}'"
                    )

            if not id_pose_map:
                self.node.get_logger().warning("No valid motor IDs found")
                return origin_map

            # 调用bus_normalize_callback处理ID-pose映射
            normalized_map = self.bus_normalize_callback(id_pose_map)

            # 将normalized_map的结果转换回电机名称并更新origin_map
            result_map = {}
            for motor_id, normalized_value in normalized_map.items():
                # 查找对应的电机名称
                motor_name = None
                for name, id_val in name_to_id_map.items():
                    if id_val == motor_id:
                        motor_name = name
                        break

                if motor_name:
                    # 更新origin_map中对应电机的值
                    motor_key = f"{motor_name}.pos"
                    result_map[motor_key] = normalized_value
                    print(
                        f"Updated {motor_key}: {normalized_value}"
                    )
                else:
                    self.node.get_logger().warning(
                        f"Could not find motor name for ID {motor_id}"
                    )

            return result_map

        except json.JSONDecodeError as e:
            self.node.get_logger().error(f"Failed to parse JSON: {e}")
            return {}
        except Exception as e:
            self.node.get_logger().error(f"Error in _cmd_string_convert_action: {e}")
            return {}


def main(args=None):
    rclpy.init(args=args)

    try:
        # 创建节点
        motor_executor = MotorExecutorNode()

        # 由于spin已经在内部线程中运行，这里只需要保持主线程活跃
        while rclpy.ok():
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 清理
        if "motor_executor" in locals():
            motor_executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
