import time
import tqdm
import numpy as np

from envs.HarfangEnv_GYM import dogfight_client as df
from envs.HarfangEnv_GYM.HarfangEnv_GYM import HarfangEnv, HarfangLowBloodEnvNew
from harfang_utils.plot import plot_2d_trajectories, plot_3d_trajectories, plot_distance
from harfang_utils.config import IP
import pickle

def get_rollouts_harfang(env, env_type, agent_group, num_games, agent_idx, final_state=False):

    df.connect(IP, 11111)  # 连接环境
    time.sleep(2)  # 等待连接
    df.disable_log()  # 关闭日志
    df.set_renderless_mode(True)  # 关闭渲染模式
    df.set_client_update_mode(True) # 更新模式

    planes = df.get_planes_list()
    plane_id = planes[0]
    oppo_id = planes[3]

    env = HarfangLowBloodEnvNew()
    env.Plane_ID_ally = plane_id  # ally 1
    env.Plane_ID_oppo = oppo_id  # ennemy_2

    oppo_state_dim = 6
    oppo_action_dim = 4
    agent_state_dim = 13
    agent_action_dim = 4
    delt_list = []

    episode = 1
    step = 0
    invalid_data = 0

    trajectories = {
        # With shape (n_timesteps, game_len), where game_len might vary across games:
        "ep_observations": [],
        "ep_actions": [],
        "ep_rewards": [], # Individual dense (= sparse + shaped * rew_shaping) reward values
        "ep_dones": [], # Individual done values

        # With shape (n_episodes, ):
        "ep_returns": [], # Sum of dense and sparse rewards across each episode
        "ep_returns_sparse": [], # Sum of sparse rewards across each episode
        "ep_lengths": [], # Lengths of each episode
    }
    
    for _ in tqdm.trange(num_games):
        success, state_list, action_list, reward_list, done_list, time_taken, tot_rews_sparse = run_agents_harfang(env, env_type, agent_group, agent_idx, include_final_state=final_state, plane_id=plane_id)
        if not success:
            continue
        obs, actions, rews, dones = state_list.T, action_list.T, reward_list.T, done_list.T
        trajectories["ep_observations"].append(obs)
        trajectories["ep_actions"].append(actions)
        trajectories["ep_rewards"].append(rews)
        trajectories["ep_dones"].append(dones)
        trajectories["ep_returns"].append(tot_rews_sparse)
        trajectories["ep_returns_sparse"].append(tot_rews_sparse)
        trajectories["ep_lengths"].append(time_taken)

    df.set_client_update_mode(False)
    df.disconnect()

    trajectories = {k: np.array(v) for k, v in trajectories.items()}
    return trajectories
    # trajectories_converted = {}
    # for k, v in trajectories.items():
    #     if k in ["ep_observations", "ep_actions", "ep_rewards", "ep_dones"]:
    #         # 这些列表包含不同长度的轨迹，需要使用 dtype=object
    #         trajectories_converted[k] = np.array(v, dtype=object)
    #     else:
    #         # 标量值可以直接转换
    #         trajectories_converted[k] = np.array(v)
    # return trajectories_converted


def run_agents_harfang(env, env_type, agent_group, agent_idx, include_final_state, **args):
    
    env.random_reset()
    df.activate_IA(args["plane_id"])
    health = 100

    EP_LEN = 3800
    ACT_DIM = 4
    done = False
    t = 0

    episode_step = 0
    lock = 0
    fire = 0
    temp_state_list = []
    temp_next_state_list = []
    temp_action_list = []
    temp_reward_list = []
    done = False

    ally_pos = []
    ennemy_pos = []
    distance = []
    missile = []
    fires = []
    locks = []

    # trajectory = []
    state_list = []
    action_list = []
    reward_list = []
    done_list = []

    if env_type == "Harfang":
        cumulative_sparse_rewards = [0.]

    while health > 0 and not done:
        health = env._get_health()
        oppo_state = env.get_oppo_observation()
        agent_state = env._get_observation()
        uni_state = [oppo_state, agent_state]
        temp_state_array = np.asarray(uni_state, dtype=object)
        temp_state_list.append(temp_state_array)
        env.set_enemy_ai()  # 在这里面设置成 AI

        if agent_state[7] > 0 and agent_state[8] > 0:
                env.fire()
                if lock == 0:
                    lock = episode_step
                    print("can step:{}".format(episode_step))

        df.update_scene()
        oppo_action = env._get_oppo_action()
        agent_action = env._get_action()
        uni_action = [oppo_action, agent_action]
        temp_action_list.append(uni_action)

        oppo_next_state = env.get_oppo_observation() 
        agent_next_state = env._get_only_observation()
        uni_next_state = [oppo_next_state, agent_next_state]
        temp_next_state_array = np.asarray(uni_next_state, dtype=object)
        temp_next_state_list.append(temp_next_state_array)

        reward = env.get_only_reward(agent_action, agent_state)
        oppo_reward = env.get_oppo_reward()
        uni_reward = [oppo_reward, reward]
        temp_reward_list.append(uni_reward)

        # 为了获取 n_oppo_next   n_agent_next

        pos = env.get_pos()
        ally_pos.append(pos)
        pos = env.get_oppo_pos()
        ennemy_pos.append(pos)

        distance.append(env.get_loc_diff(agent_state))
        # if env.now_missile_state:
        #     fires.append(episode_step)
        if env.missile1_state:
            missile.append(episode_step)
        if env.Ally_target_locked:
            locks.append(episode_step)

        if agent_action[-1] > 0:
            # print('fire step:{}'.format(episode_step))
            if fire == 0:
                fire = episode_step
            fires.append(episode_step)

        episode_step += 1

        cumulative_sparse_rewards[0] += reward
        
        if episode_step > 4000:
            state_list.append(agent_state)
            action_list.append(agent_action)
            reward_list.append(reward)
            done_list.append(True)
            done = True
        else:
            state_list.append(agent_state)
            action_list.append(agent_action)
            reward_list.append(reward)
            done_list.append(False)
            done = False
    
    if episode_step > 4000 or fire - lock > 20 or fire - lock < 0:
            return False, np.array(state_list), np.array(action_list), np.array(reward_list), np.array(done_list), None, None
    else:
        return True, np.array(state_list), np.array(action_list), np.array(reward_list), np.array(done_list), t, np.array(cumulative_sparse_rewards)