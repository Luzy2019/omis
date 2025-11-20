import numpy as np
import environments.dogfight_client as df
from environments.constants import *
import gym
import os
import inspect
import random
import math


class HarfangEnv:
    def __init__(self):
        self.done = False  # 游戏是否结束的标志
        self.loc_diff = 0  # 位置差异（可能用于计算奖励或状态）
        # 定义动作空间：4个连续动作，每个动作取值范围为[-1.0, 1.0]
        # 通常对应：俯仰、滚转、偏航、油门控制
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64,
        )
        self.Plane_ID_oppo = "ennemy_2"  # 敌方飞机名称
        self.Plane_ID_ally = "ally_1"  # 我方飞机名称
        self.Aircraft_Loc = None  # 飞机位置
        self.Ally_target_locked = False  # 动作执行前是否锁定敌方
        self.n_Ally_target_locked = False  # 动作执行后是否锁定敌方
        self.reward = 0  # 奖励值
        self.Plane_Irtifa = 0  # Irtifa飞机

        self.now_missile_state = False  # 导弹此时是否发射（本次step是否发射了导弹）
        self.missile1_state = True  # 动作执行前导弹1是否存在
        self.n_missile1_state = True  # 动作执行后导弹1是否存在
        self.missile = df.get_machine_missiles_list(self.Plane_ID_ally)  # 导弹列表
        self.missile1_id = self.missile[0]  # 导弹1

        self.oppo_health = 0.2  # 敌方飞机血量
        self.target_angle = None  # 目标角度

        self.success = 0  # 步骤成功
        self.episode_success = False  # 是否成功完成一个回合
        self.fire_success = False  # 是否成功发射导弹

    def reset(self):  # reset simulation beginning of episode
        self.Ally_target_locked = False  # 动作执行前是否锁定敌方
        self.n_Ally_target_locked = False  # 动作执行后是否锁定敌方
        self.missile1_state = True  # 动作执行前导弹1是否存在
        self.n_missile1_state = True  # 动作执行后导弹1是否存在
        self.success = 0
        self.done = False
        self._reset_machine()
        self._reset_missile()  # 重设导弹
        state_ally = self._get_observation()  # get observations
        df.set_target_id(
            self.Plane_ID_ally, self.Plane_ID_oppo
        )  # set target, for firing missile
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally  # 当前时刻状态

        return state_ally

    def random_reset(self):  # reset simulation beginning of episode
        self.Ally_target_locked = False  # 运用动作前是否锁敌
        self.n_Ally_target_locked = False  # 运用动作后是否锁敌
        self.missile1_state = True  # 运用动作前导弹1是否存在
        self.n_missile1_state = True  # 运用动作后导弹1是否存在
        self.success = 0
        self.done = False
        self._random_reset_machine()
        self._reset_missile()  # 重设导弹
        state_ally = self._get_observation()  # get observations
        df.set_target_id(
            self.Plane_ID_ally, self.Plane_ID_oppo
        )  # set target, for firing missile
        self.episode_success = False
        self.fire_success = False
        self.state = state_ally

        return state_ally

    def _random_reset_machine(self):
        df.reset_machine("ally_1")  # 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)  # 设置的为健康水平，即血量/100
        self.oppo_health = 0.2  #
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100),
            0,
            0,
            0,
        )

        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def step_test(self, action):
        self._apply_action(action)  # apply neural networks output
        n_state = self._get_observation()  # 执行动作后的状态
        self._get_reward(self.state, action, n_state)  # get reward value
        self.state = n_state
        self._get_termination()  # check termination conditions

        return (
            n_state,
            self.reward,
            self.done,
            {},
            self.now_missile_state,
            self.missile1_state,
            self.n_missile1_state,
            self.Ally_target_locked,
            self.episode_success,
        )

    def _get_reward(self, state, action, n_state):
        self.reward = 0
        self.success = 0
        self._get_loc_diff()  # get location difference information for reward

        # 距离惩罚：帮助追击
        self.reward -= 0.0001 * (self.loc_diff)  # 0.4

        # 目标角惩罚：帮助锁敌
        self.reward -= (self.target_angle) * 10

        # if self.Plane_Irtifa < 2000:
        #     self.reward -= 4

        # if self.Plane_Irtifa > 7000:
        #     self.reward -= 4

        # 开火奖励：帮助开火
        if (
            self.now_missile_state == True
        ):  # 如果此step导弹发射   必须保证锁敌的情况下才能发射  验证锁敌情况  hmy
            self.reward += 8
            if (
                self.missile1_state == True and self.Ally_target_locked == False
            ):  # 且导弹存在、不锁敌
                # self.reward -= 100 # 4、4
                # self.success = -1
                self.success = 0 # 不成功
                print("failed to fire")
            elif (
                self.missile1_state == True and self.Ally_target_locked == True
            ):  # 且导弹存在且锁敌
                # self.reward += 8 # 100、4
                print("successful to fire")
                self.success = 1
                self.fire_success = True
            else:  # 导弹不存在了 hmy
                # self.reward -= 10
                self.reward -= 0

        # 坠落奖励（最终奖励）
        if self.oppo_health["health_level"] <= 0.1 and self.fire_success:
            # self.reward += 600 # 无、无
            print("enemy have fallen")

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(0))

        # self.now_missile_state = False

        if float(action_ally[3] > 0):  # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
            df.fire_missile(self.Plane_ID_ally, 0)  #
            self.now_missile_state = True  # 此时导弹发射
            # print("fire")
        else:
            self.now_missile_state = False

        df.update_scene()

    def _get_termination(self):  # 测试时候应用
        # if self.loc_diff < 200:
        #     self.done = True
        if self.Plane_Irtifa < 200 or self.Plane_Irtifa > 10000:  # try  修改一下？
            self.done = True
        if self.oppo_health["health_level"] <= 0.05:  # 敌机血量低于0则结束
            self.done = True
            self.episode_success = True
        # if self.now_missile_state == True:
        #     self.done = True

    def _reset_machine(self):
        df.reset_machine("ally_1")  # 初始化两个飞机
        df.reset_machine("ennemy_2")
        df.set_health("ennemy_2", 0.2)  # 设置的为健康水平，即血量/100
        self.oppo_health = 0.2  #
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)

        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_ally)
        df.retract_gear(self.Plane_ID_oppo)

    def _reset_missile(self):  #
        self.now_missile_state = False
        df.rearm_machine(self.Plane_ID_ally)  # 重新装填导弹

    def _get_loc_diff(self):
        self.loc_diff = (
            ((self.Aircraft_Loc[0] - self.Oppo_Loc[0]) ** 2)
            + ((self.Aircraft_Loc[1] - self.Oppo_Loc[1]) ** 2)
            + ((self.Aircraft_Loc[2] - self.Oppo_Loc[2]) ** 2)
        ) ** (1 / 2)

    def _get_observation(self):  # 注意get的是n_state
        # Plane States
        Plane_state = df.get_plane_state(self.Plane_ID_ally)
        Plane_Pos = [
            Plane_state["position"][0] / NormStates["Plane_position"],
            Plane_state["position"][1] / NormStates["Plane_position"],
            Plane_state["position"][2] / NormStates["Plane_position"],
        ]
        Plane_Euler = [
            Plane_state["Euler_angles"][0]
            / NormStates["Plane_Euler_angles"],  # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
            Plane_state["Euler_angles"][1]
            / NormStates["Plane_Euler_angles"],  # 航向角，0 -> pai -> -pai -> 0
            Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]  # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Plane_Heading = (
            Plane_state["heading"] / NormStates["Plane_heading"]
        )  # 航向角，0 -> 360
        Plane_Pitch_Att = (
            Plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        )  # 俯仰：俯0 -> -90，仰0 -> 90
        Plane_Roll_Att = (
            Plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"]
        )  # 横滚角，顺时针：0 -> -90 ->0 -> 90 -> 0

        Plane_Pitch_Level = Plane_state["user_pitch_level"]
        Plane_Yaw_Level = Plane_state["user_yaw_level"]
        Plane_Roll_Level = Plane_state["user_roll_level"]

        # Opponent States
        Oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        Oppo_Pos = [
            Oppo_state["position"][0] / NormStates["Plane_position"],
            Oppo_state["position"][1] / NormStates["Plane_position"],
            Oppo_state["position"][2] / NormStates["Plane_position"],
        ]
        Oppo_Euler = [
            Oppo_state["Euler_angles"][0]
            / NormStates["Plane_Euler_angles"],  # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
            Oppo_state["Euler_angles"][1]
            / NormStates["Plane_Euler_angles"],  # 航向角，0 -> pai -> -pai -> 0
            Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]  # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Oppo_Pitch_Att = (
            Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        )
        Oppo_Roll_Att = Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"]

        self.Plane_Irtifa = Plane_state["position"][1]
        self.Aircraft_Loc = Plane_state["position"]
        self.Oppo_Loc = Oppo_state["position"]

        self.Ally_target_locked = self.n_Ally_target_locked
        self.n_Ally_target_locked = Plane_state["target_locked"]
        if self.n_Ally_target_locked == True:  #
            locked = 1
        else:
            locked = -1

        target_angle = Plane_state["target_angle"] / 180
        self.target_angle = target_angle

        Pos_Diff = [
            Plane_Pos[0] - Oppo_Pos[0],
            Plane_Pos[1] - Oppo_Pos[1],
            Plane_Pos[2] - Oppo_Pos[2],
        ]

        self.oppo_health = df.get_health(self.Plane_ID_oppo)

        oppo_hea = self.oppo_health["health_level"]  # 敌机初始血量为20

        # if self.now_missile_state == True:
        #     if_fire = 1
        # else:
        #     if_fire = -1

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)

        self.missile1_state = self.n_missile1_state
        self.n_missile1_state = Missile_state["missiles_slots"][0]  # 更新导弹1是否存在
        if self.n_missile1_state == True:
            missile1_state = 1
        else:
            missile1_state = -1

        # States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
        #  Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, missile1_state), axis=None) # 感觉加入敌机健康值没用

        # States = np.concatenate((Plane_Pos, Plane_Euler, Plane_Pitch_Level, Plane_Yaw_Level, Plane_Roll_Level, target_angle, locked, missile1_state, Oppo_Pos, Oppo_Euler, oppo_hea), axis=None)

        States = np.concatenate(
            (
                Pos_Diff,
                Plane_Euler,
                target_angle,
                locked,
                missile1_state,
                Plane_Pos,
                oppo_hea,
            ),
            axis=None,
        )

        # 相对位置（3），我机欧拉角（3），锁敌角，是否锁敌，导弹状态，敌机欧拉角（3），敌机血量 , 绝对位置（3）(用绝对位置替代敌机欧拉角)

        # self.now_missile_state = False # 未来的状态均不发射（不能加，因为后续计算奖励函数需要）

        return States

    def _get_only_observation(self):  # 注意get的是n_state
        # Plane States
        Plane_state = df.get_plane_state(self.Plane_ID_ally)
        Plane_Pos = [
            Plane_state["position"][0] / NormStates["Plane_position"],
            Plane_state["position"][1] / NormStates["Plane_position"],
            Plane_state["position"][2] / NormStates["Plane_position"],
        ]
        Plane_Euler = [
            Plane_state["Euler_angles"][0]
            / NormStates["Plane_Euler_angles"],  # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
            Plane_state["Euler_angles"][1]
            / NormStates["Plane_Euler_angles"],  # 航向角，0 -> pai -> -pai -> 0
            Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]  # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Plane_Heading = (
            Plane_state["heading"] / NormStates["Plane_heading"]
        )  # 航向角，0 -> 360
        Plane_Pitch_Att = (
            Plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        )  # 俯仰：俯0 -> -90，仰0 -> 90
        Plane_Roll_Att = (
            Plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"]
        )  # 横滚角，顺时针：0 -> -90 ->0 -> 90 -> 0

        Plane_Pitch_Level = Plane_state["user_pitch_level"]
        Plane_Yaw_Level = Plane_state["user_yaw_level"]
        Plane_Roll_Level = Plane_state["user_roll_level"]

        # Opponent States
        Oppo_state = df.get_plane_state(self.Plane_ID_oppo)
        Oppo_Pos = [
            Oppo_state["position"][0] / NormStates["Plane_position"],
            Oppo_state["position"][1] / NormStates["Plane_position"],
            Oppo_state["position"][2] / NormStates["Plane_position"],
        ]
        Oppo_Euler = [
            Oppo_state["Euler_angles"][0]
            / NormStates["Plane_Euler_angles"],  # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
            Oppo_state["Euler_angles"][1]
            / NormStates["Plane_Euler_angles"],  # 航向角，0 -> pai -> -pai -> 0
            Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]  # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Oppo_Pitch_Att = (
            Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        )
        Oppo_Roll_Att = Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"]

        if Plane_state["target_locked"] == True:  #
            locked = 1
        else:
            locked = -1

        target_angle = Plane_state["target_angle"] / 180

        Pos_Diff = [
            Plane_Pos[0] - Oppo_Pos[0],
            Plane_Pos[1] - Oppo_Pos[1],
            Plane_Pos[2] - Oppo_Pos[2],
        ]

        oppo_hea = df.get_health(self.Plane_ID_oppo)  # 敌机初始血量为20

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)

        if Missile_state["missiles_slots"][0] == True:
            missile1_state = 1
        else:
            missile1_state = -1

        # States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
        #  Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, missile1_state), axis=None) # 感觉加入敌机健康值没用

        # States = np.concatenate((Plane_Pos, Plane_Euler, Plane_Pitch_Level, Plane_Yaw_Level, Plane_Roll_Level, target_angle, locked, missile1_state, Oppo_Pos, Oppo_Euler, oppo_hea), axis=None)

        States = np.concatenate(
            (
                Pos_Diff,
                Plane_Euler,
                target_angle,
                locked,
                missile1_state,
                Plane_Pos,
                oppo_hea["health_level"],
            ),
            axis=None,
        )

        # 相对位置（3），我机欧拉角（3），锁敌角，是否锁敌，导弹状态，敌机欧拉角（3），敌机血量

        # 我机位置（3），我机欧拉角（3），三翼角度（3），锁敌角，是否锁敌，导弹状态，敌机位置（3），敌机欧拉角（3），敌机血量

        # 距离差距(3), 飞机欧拉角(3), 飞机航向角, 敌机航向角, 敌机俯仰, 敌机滚动, 锁敌角, 敌机血量, 是否锁敌, 导弹状态

        # self.now_missile_state = False # 未来的状态均不发射（不能加，因为后续计算奖励函数需要）

        return States

    def get_oppo_observation(self):  # 注意get的是n_state
        # Opponent States
        Plane_state = df.get_plane_state(self.Plane_ID_oppo)
        Plane_Pos = [
            Plane_state["position"][0] / NormStates["Plane_position"],
            Plane_state["position"][1] / NormStates["Plane_position"],
            Plane_state["position"][2] / NormStates["Plane_position"],
        ]
        Plane_Euler = [
            Plane_state["Euler_angles"][0]
            / NormStates["Plane_Euler_angles"],  # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
            Plane_state["Euler_angles"][1]
            / NormStates["Plane_Euler_angles"],  # 航向角，0 -> pai -> -pai -> 0
            Plane_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]  # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Plane_Heading = (
            Plane_state["heading"] / NormStates["Plane_heading"]
        )  # 航向角，0 -> 360
        Plane_Pitch_Att = (
            Plane_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        )  # 俯仰：俯0 -> -90，仰0 -> 90
        Plane_Roll_Att = (
            Plane_state["roll_attitude"] / NormStates["Plane_roll_attitude"]
        )  # 横滚角，顺时针：0 -> -90 ->0 -> 90 -> 0

        Plane_Pitch_Level = Plane_state["user_pitch_level"]
        Plane_Yaw_Level = Plane_state["user_yaw_level"]
        Plane_Roll_Level = Plane_state["user_roll_level"]

        # Ally States
        Oppo_state = df.get_plane_state(self.Plane_ID_ally)
        Oppo_Pos = [
            Oppo_state["position"][0] / NormStates["Plane_position"],
            Oppo_state["position"][1] / NormStates["Plane_position"],
            Oppo_state["position"][2] / NormStates["Plane_position"],
        ]
        Oppo_Euler = [
            Oppo_state["Euler_angles"][0]
            / NormStates["Plane_Euler_angles"],  # 俯仰：俯0 -> pai/2，仰0 -> -pai/2
            Oppo_state["Euler_angles"][1]
            / NormStates["Plane_Euler_angles"],  # 航向角，0 -> pai -> -pai -> 0
            Oppo_state["Euler_angles"][2] / NormStates["Plane_Euler_angles"],
        ]  # 横滚角，顺时针：0 -> -pai -> pai -> 0
        Oppo_Heading = Oppo_state["heading"] / NormStates["Plane_heading"]
        Oppo_Pitch_Att = (
            Oppo_state["pitch_attitude"] / NormStates["Plane_pitch_attitude"]
        )
        Oppo_Roll_Att = Oppo_state["roll_attitude"] / NormStates["Plane_roll_attitude"]

        # self.Plane_Irtifa = Plane_state["position"][1]
        # self.Aircraft_Loc = Plane_state["position"]
        # self.Oppo_Loc = Oppo_state["position"]

        # self.Ally_target_locked = self.n_Ally_target_locked
        # self.n_Ally_target_locked = Plane_state["target_locked"]
        if Plane_state["target_locked"] == True:  #
            locked = 1
        else:
            locked = -1

        target_angle = Plane_state["target_angle"] / 180
        # self.target_angle = target_angle

        Pos_Diff = [
            Plane_Pos[0] - Oppo_Pos[0],
            Plane_Pos[1] - Oppo_Pos[1],
            Plane_Pos[2] - Oppo_Pos[2],
        ]

        ally_health = df.get_health(self.Plane_ID_ally)

        oppo_hea = ally_health["health_level"]

        # if self.now_missile_state == True:
        #     if_fire = 1
        # else:
        #     if_fire = -1

        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_oppo)

        # self.missile1_state = self.n_missile1_state
        # self.n_missile1_state = Missile_state["missiles_slots"][0]  # 更新导弹1是否存在
        if Missile_state["missiles_slots"][0] == True:
            missile1_state = 1
        else:
            missile1_state = -1

        # States = np.concatenate((Pos_Diff, Plane_Euler, Plane_Heading,
        #  Oppo_Heading, Oppo_Pitch_Att, Oppo_Roll_Att, target_angle, oppo_hea, locked, missile1_state), axis=None) # 感觉加入敌机健康值没用

        # States = np.concatenate((Plane_Pos, Plane_Euler, Plane_Pitch_Level, Plane_Yaw_Level, Plane_Roll_Level, target_angle, locked, missile1_state, Oppo_Pos, Oppo_Euler, oppo_hea), axis=None)

        States = np.concatenate(
            (Oppo_Pos, Oppo_Euler), axis=None
        )  # 将相对位置换成绝对位置

        # 相对位置（3），我机欧拉角（3），锁敌角，是否锁敌，导弹状态，敌机欧拉角（3），敌机血量

        # 我机位置（3），我机欧拉角（3），三翼角度（3），锁敌角，是否锁敌，导弹状态，敌机位置（3），敌机欧拉角（3），敌机血量

        # 距离差距(3), 飞机欧拉角(3), 飞机航向角, 敌机航向角, 敌机俯仰, 敌机滚动, 锁敌角, 敌机血量, 是否锁敌, 导弹状态

        # self.now_missile_state = False # 未来的状态均不发射（不能加，因为后续计算奖励函数需要）

        return States

    def get_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_ally)
        return np.array(
            [
                plane_state["position"][0],
                plane_state["position"][1],
                plane_state["position"][2],
            ]
        )

    def get_oppo_pos(self):
        plane_state = df.get_plane_state(self.Plane_ID_oppo)
        return np.array(
            [
                plane_state["position"][0],
                plane_state["position"][1],
                plane_state["position"][2],
            ]
        )

    def save_parameters_to_txt(self, log_dir):
        # os.makedirs(log_dir)
        source_code1 = inspect.getsource(self._get_reward)
        source_code2 = inspect.getsource(self._reset_machine)
        source_code3 = inspect.getsource(self._get_termination)

        filename = os.path.join(log_dir, "log2.txt")
        with open(filename, "w") as file:
            file.write(source_code1)
            file.write(" ")
            file.write(source_code2)
            file.write(" ")
            file.write(source_code3)

    # for expert data

    def get_loc_diff(self, state):
        loc_diff = (
            (((state[0]) * 10000) ** 2)
            + (((state[1]) * 10000) ** 2)
            + (((state[2]) * 10000) ** 2)
        ) ** (1 / 2)
        return loc_diff

    # 改成 -4
    def get_termination(self, state):
        done = False
        if state[-1] <= 0.1:  # 敌机血量低于0则结束
            done = True
        return done

    # 修改action
    def _get_action(self):
        action = [0, 0, 0, -1]
        state = df.get_plane_state(self.Plane_ID_ally)
        # print(state)
        # {'timestamp': 70310, 'timestep': 0.016666666666666666, 'position': [-16.567607879638672, 149.64927673339844, 183.3050537109375], 'Euler_angles': [-0.7854030728340149, 0.0050728581845760345, -0.12579230964183807], 'easy_steering': True, 'health_level': 1, 'destroyed': False, 'wreck': False, 'crashed': False, 'active': True, 'type': 'AICRAFT', 'nationality': 1, 'thrust_level': 1, 'brake_level': 0, 'flaps_level': 0, 'horizontal_speed': 112.246337890625, 'vertical_speed': 97.5827865600586, 'linear_speed': 148.73345947265625, 'move_vector': [-11.339219093322754, 97.5827865600586, 111.672119140625], 'linear_acceleration': 1.4310089111328068, 'altitude': 149.64927673339844, 'heading': 0.29141239305266703, 'pitch_attitude': 45.00029076022247, 'roll_attitude': -7.207452358089359, 'post_combustion': True, 'user_pitch_level': 2.4729337383178063e-05, 'user_roll_level': -0.06377019733190536, 'user_yaw_level': 0.0, 'gear': False, 'ia': True, 'autopilot': True, 'autopilot_heading': 1.1798513347878505, 'autopilot_speed': -1, 'autopilot_altitude': 1500, 'target_id': 'ennemy_2', 'target_locked': False, 'target_out_of_range': True, 'target_angle': 6.8759194555908145}
        Pitch_Att = state["user_pitch_level"]
        Roll_Att = state["user_roll_level"]
        Yaw_Att = state["user_yaw_level"]
        action[0] = Pitch_Att
        action[1] = Roll_Att
        action[2] = Yaw_Att
        Missile_state = df.get_missiles_device_slots_state(self.Plane_ID_ally)
        self.n_missile1_state = Missile_state["missiles_slots"][0]

        if self.missile1_state == True and self.n_missile1_state == False:
            action[3] = 1
        if self.n_missile1_state == False and state["target_locked"] == True:
            action[3] = 1
        # if self.n_missile1_state == False:
        #     action[3] = 1
        # 改成锁定即为1
        return action

    def _get_oppo_action(self):
        action = [0, 0, 0, -1]
        state = df.get_plane_state(self.Plane_ID_oppo)
        # print(state)
        # {'timestamp': 70310, 'timestep': 0.016666666666666666, 'position': [-16.567607879638672, 149.64927673339844, 183.3050537109375], 'Euler_angles': [-0.7854030728340149, 0.0050728581845760345, -0.12579230964183807], 'easy_steering': True, 'health_level': 1, 'destroyed': False, 'wreck': False, 'crashed': False, 'active': True, 'type': 'AICRAFT', 'nationality': 1, 'thrust_level': 1, 'brake_level': 0, 'flaps_level': 0, 'horizontal_speed': 112.246337890625, 'vertical_speed': 97.5827865600586, 'linear_speed': 148.73345947265625, 'move_vector': [-11.339219093322754, 97.5827865600586, 111.672119140625], 'linear_acceleration': 1.4310089111328068, 'altitude': 149.64927673339844, 'heading': 0.29141239305266703, 'pitch_attitude': 45.00029076022247, 'roll_attitude': -7.207452358089359, 'post_combustion': True, 'user_pitch_level': 2.4729337383178063e-05, 'user_roll_level': -0.06377019733190536, 'user_yaw_level': 0.0, 'gear': False, 'ia': True, 'autopilot': True, 'autopilot_heading': 1.1798513347878505, 'autopilot_speed': -1, 'autopilot_altitude': 1500, 'target_id': 'ennemy_2', 'target_locked': False, 'target_out_of_range': True, 'target_angle': 6.8759194555908145}
        Pitch_Att = state["user_pitch_level"]
        Roll_Att = state["user_roll_level"]
        Yaw_Att = state["user_yaw_level"]
        action[0] = Pitch_Att
        action[1] = Roll_Att
        action[2] = Yaw_Att
        return action

    def _get_health(self):
        state = df.get_plane_state(self.Plane_ID_oppo)
        health = state["health_level"]
        return health

    def get_only_reward(self, action, n_state):
        reward = 0
        step_success = 0
        loc_diff = self.get_loc_diff(
            n_state
        )  # get location difference information for reward

        # 距离惩罚：帮助追击
        reward -= 0.0001 * loc_diff

        # 目标角惩罚：帮助锁敌
        reward -= (n_state[6]) * 10

        # 开火奖励：帮助开火  锁敌奖励
        if action[-1] > 0:  # 如果导弹发射
            # reward -= 8
            reward += 8
        # 击落奖励待修改

        return reward

    def get_oppo_reward(self):
        reward = 0.5
        return reward

    def fire(self):
        df.fire_missile(self.Plane_ID_ally, 0)

    # for collect data
    def get_success(self):
        return self.fire_success


class HarfangSerpentineEnv(HarfangEnv):
    def __init__(self):
        super(HarfangSerpentineEnv, self).__init__()

    def get_oppo_reward(self):
        reward = 0.3
        return reward

    def set_ennemy_yaw(self):
        self.serpentine_step += 1

        if self.serpentine_step % self.duration == 0:
            self.serpentine_step = 0
            # 切换偏航方向（正负交替）
            self.oppo_yaw = 0.1 * (-1 if self.oppo_yaw > 0 else 1)
            self.duration = 500  # 300  #  两个duration

        df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0))
        df.set_plane_yaw(self.Plane_ID_oppo, float(self.oppo_yaw))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        self.set_ennemy_yaw()  # oppo next action ?  oppo action

        # self.now_missile_state = False

        if float(action_ally[3] > 0):  # 大于0发射导弹
            # if self.missile1_state == True: # 如果导弹存在（不判断是为了奖励函数服务，也符合动作逻辑）
            df.fire_missile(self.Plane_ID_ally, 0)  #
            self.now_missile_state = True  # 此时导弹发射
            # print("fire")
        else:
            self.now_missile_state = False

        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2  # gai

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2  # gai

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100),
            0,
            0,
            0,
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.oppo_yaw = 0
        self.serpentine_step = 0
        # 初始偏航角
        self.oppo_yaw = -0.1
        self.duration = 250  # 150

        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2)  # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.6)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 200)
        df.retract_gear(self.Plane_ID_oppo)


class HarfangCircularEnv(HarfangEnv):
    def __init__(self):
        super(HarfangCircularEnv, self).__init__()

    # def set_ennemy_yaw(self):  # 最原始的版
    #     self.circular_step += 1

    #     if self.circular_step < 100:
    #         df.set_plane_pitch(self.Plane_ID_oppo, float(-0.02))
    #         df.set_plane_roll(self.Plane_ID_oppo, float(0.84))
    #     else:
    #         df.set_plane_pitch(self.Plane_ID_oppo, float(-0.01))
    #     df.set_plane_roll(self.Plane_ID_oppo, float(0.28))
    #     df.set_plane_yaw(self.Plane_ID_oppo, float(0))

    #     # if self.circular_step < 1000:
    #     #     df.set_plane_pitch(self.Plane_ID_ally, float(-0.02))
    #     #     df.set_plane_roll(self.Plane_ID_ally, float(0.81))
    #     # else:
    #     #     df.set_plane_pitch(self.Plane_ID_ally, float(-0.01))
    #     # df.set_plane_roll(self.Plane_ID_ally, float(0.27))
    #     # df.set_plane_yaw(self.Plane_ID_ally, float(0))
    def get_oppo_reward(self):
        reward = 0.1
        return reward

    def set_ennemy_yaw(self):
        self.circular_step += 1

        if self.circular_step < 100:
            df.set_plane_pitch(self.Plane_ID_oppo, float(-0.2))
        else:
            df.set_plane_pitch(self.Plane_ID_oppo, float(0))
        df.set_plane_roll(self.Plane_ID_oppo, float(0.15))
        df.set_plane_yaw(self.Plane_ID_oppo, float(-0.1))

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        self.set_ennemy_yaw()

        if float(action_ally[3] > 0):  # 大于0发射导弹
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True  # 此时导弹发射
            # print("fire1")
        else:
            self.now_missile_state = False

        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = 0.2

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100),
            0,
            0,
            0,
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        self.circular_step = 0

        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, 0.2)  # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.8)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 290)
        df.retract_gear(self.Plane_ID_oppo)


class HarfangLowBloodEnvNew(HarfangEnv):
    def __init__(self):
        super(HarfangLowBloodEnvNew, self).__init__()
        self.init_health_level = 0.20

    def set_enemy_ai(self):
        df.fire_missile(self.Plane_ID_oppo, 0)
        df.fire_missile(self.Plane_ID_oppo, 1)
        df.fire_missile(self.Plane_ID_oppo, 2)
        df.fire_missile(self.Plane_ID_oppo, 3)
        df.activate_IA(self.Plane_ID_oppo)

    def get_oppo_reward(self):
        reward = 0.05
        return reward

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.activate_IA(self.Plane_ID_oppo)

        if float(action_ally[3] > 0):  # 大于0发射导弹
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True  # 此时导弹发射
        else:
            self.now_missile_state = False

        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = self.init_health_level

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = self.init_health_level

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100),
            0,
            0,
            0,
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        df.fire_missile(self.Plane_ID_oppo, 0)
        df.fire_missile(self.Plane_ID_oppo, 1)
        df.fire_missile(self.Plane_ID_oppo, 2)
        df.fire_missile(self.Plane_ID_oppo, 3)
        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, self.init_health_level)  # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.8)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 290)
        df.retract_gear(self.Plane_ID_oppo)


class HarfangNormalBloodEnvNew(HarfangEnv):
    def __init__(self):
        super(HarfangNormalBloodEnvNew, self).__init__()
        self.init_health_level = 0.21

    def set_enemy_ai(self):
        df.fire_missile(self.Plane_ID_oppo, 0)
        df.fire_missile(self.Plane_ID_oppo, 1)
        df.fire_missile(self.Plane_ID_oppo, 2)
        df.fire_missile(self.Plane_ID_oppo, 3)
        df.activate_IA(self.Plane_ID_oppo)

    def get_oppo_reward(self):
        reward = 0.05
        return reward

    def _apply_action(self, action_ally):
        df.set_plane_pitch(self.Plane_ID_ally, float(action_ally[0]))
        df.set_plane_roll(self.Plane_ID_ally, float(action_ally[1]))
        df.set_plane_yaw(self.Plane_ID_ally, float(action_ally[2]))

        df.activate_IA(self.Plane_ID_oppo)

        if float(action_ally[3] > 0):  # 大于0发射导弹
            df.fire_missile(self.Plane_ID_ally, 0)
            self.now_missile_state = True  # 此时导弹发射
        else:
            self.now_missile_state = False

        df.update_scene()

    def _reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = self.init_health_level

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(self.Plane_ID_ally, 0, 3500, -4000, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def _random_reset_machine(self):
        self.reset_ennemy()
        self.oppo_health = self.init_health_level

        df.reset_machine(self.Plane_ID_ally)
        df.reset_machine_matrix(
            self.Plane_ID_ally,
            0 + random.randint(-100, 100),
            3500 + random.randint(-100, 100),
            -4000 + random.randint(-100, 100),
            0,
            0,
            0,
        )
        df.set_plane_thrust(self.Plane_ID_ally, 1)
        df.set_plane_linear_speed(self.Plane_ID_ally, 300)
        df.retract_gear(self.Plane_ID_ally)

    def reset_ennemy(self):
        df.fire_missile(self.Plane_ID_oppo, 0)
        df.fire_missile(self.Plane_ID_oppo, 1)
        df.fire_missile(self.Plane_ID_oppo, 2)
        df.fire_missile(self.Plane_ID_oppo, 3)
        df.reset_machine(self.Plane_ID_oppo)
        df.set_health(self.Plane_ID_oppo, self.init_health_level)  # 设置的为健康水平，即血量/100
        df.reset_machine_matrix(self.Plane_ID_oppo, 0, 4200, 0, 0, 0, 0)
        df.set_plane_thrust(self.Plane_ID_oppo, 0.8)
        df.set_plane_linear_speed(self.Plane_ID_oppo, 290)
        df.retract_gear(self.Plane_ID_oppo)
