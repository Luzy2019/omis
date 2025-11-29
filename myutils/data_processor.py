import pandas as pd
import numpy as np
import os
import pickle
import csv
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
LOG = logging.getLogger()

def load_agent_oppo_data(data_path, agent_index, oppo_index, act_dim, config_dict):

    config_dict["NUM_OPPO_POLICY"] = len(data_path)
    num_agent_policy = 1
    num_oppo_policy = len(data_path)
    returns_against_oppo_list = [[[] for __ in range(num_agent_policy)] for _ in range(num_oppo_policy)]
    
    data_list = [[] for _ in range(num_oppo_policy)]

    for policy in range(len(data_path)): # 1
        data_o = []         # observation
        data_a = []         # action
        data_r = []         # reward
        data_o_next = []    # next_observation
        data_done = []      # done
        pkl_files = [f for f in os.listdir(data_path[policy]) if f.endswith('.pkl')]

        for pkl_file in pkl_files:
            file_path = os.path.join(data_path[policy], pkl_file)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_o.append(data["observations"])
                data_a.append(data["actions"])
                data_r.append(data["rewards"])
                data_o_next.append(data["next_observations"])
                data_done.append(data["dones"])

        num_epis = len(data_o) # 2000
        
        for e in range(num_epis): # 2000
            num_steps = len(data_o[e]) # 3091
            for agent in agent_index: # 1
                agent_o_ep = []
                agent_a_ep = []
                agent_r_ep = []
                done = []
                for oppo in oppo_index:
                    oppo_o_ep = []
                    oppo_a_ep = []
                    oppo_r_ep = []
                    oppo_o_next_ep = []
                    for k in range(num_steps):
                        agent_o_ep.append(np.array(data_o[e][k][agent]))
                        agent_a_ep.append(np.array(data_a[e][k][agent])[:act_dim])
                        agent_r_ep.append(np.array(data_r[e][k][agent]))

                        oppo_o_ep.append(np.array(data_o[e][k][oppo]))
                        oppo_a_ep.append(np.array(data_a[e][k][oppo])[:act_dim])
                        oppo_r_ep.append(np.array(data_r[e][k][oppo]))
                        oppo_o_next_ep.append(np.array(data_o_next[e][k][oppo]))

                        done.append(np.array(data_done[e][k]))
                    data_list[policy].append([
                        {
                            "observations": np.array(agent_o_ep),
                            "actions": np.array(agent_a_ep),
                            "rewards": np.array(agent_r_ep),
                            "dones": np.array(done)
                        },
                        {
                            "observations": np.array(oppo_o_ep),
                            "actions": np.array(oppo_a_ep),
                            "rewards": np.array(oppo_r_ep),
                            "next_observations": np.array(oppo_o_next_ep),
                        }
                    ])
                    returns_against_oppo_list[policy][0].append(np.sum(agent_r_ep))
                    
    num_trajs_list = []
    oppo_baseline_list = []
    oppo_target_list = []

    for i in range(num_oppo_policy):
        num_trajs_list.append(len(data_list[i]))
        returns_against_oppo_mean = []
        for j in range(num_agent_policy):
            if returns_against_oppo_list[i][j] != []:
                returns_against_oppo_mean.append(np.mean(returns_against_oppo_list[i][j]))
        oppo_baseline_list.append(np.mean(returns_against_oppo_mean))
        oppo_target_list.append(np.max(returns_against_oppo_mean))

    config_dict["NUM_TRAJS"] = num_trajs_list
    config_dict["OPPO_BASELINE"] = oppo_baseline_list
    config_dict["OPPO_TARGET"] = oppo_target_list
    
    LOG.info(f"num_trajs_list: {num_trajs_list}")
    LOG.info(f"oppo_baseline_list: {oppo_baseline_list}")
    LOG.info(f"oppo_baseline_mean: {np.mean(oppo_baseline_list)}")
    LOG.info(f"oppo_target_list: {oppo_target_list}")
    LOG.info(f"oppo_target_mean: {np.mean(oppo_target_list)}")

    return data_list

def load_pkl_data(dir):
    states = []
    actions = []
    pkl_files = [f for f in os.listdir(dir) if f.endswith(".pkl")]
    for pkl_file in pkl_files:
        file_path = os.path.join(dir, pkl_file)
        with open(file_path, "rb") as file:
            data = pickle.load(file)

            obs = data["observations"]  # (num_samples, n)
            obs_flat = obs.reshape(-1, obs.shape[-1])  # 拉平成二维数组
            states.extend(obs_flat[:, 1])

            act = data["actions"][:, 1]  # (num_samples, n, d)
            act_flat = act.reshape(-1, act.shape[-1])  # 拉平成二维数组
            actions.extend(act_flat)

    return states, actions

def read_data(dir):
    _state, _action = load_pkl_data(dir)

    state = np.array(_state, dtype=np.float32)
    # npstate = np.array([np.fromstring(item[1:-1], sep=' ') for item in state])  # 将state变为(?, 13)的格式，一行代表一个state

    action = np.array(_action, dtype=np.float32)
    # npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])

    # return npstate, npaction
    return state, action

def write_data(npstate, npaction, data_dir):
    data = [npstate, npaction]
    with open(data_dir, 'w', newline='') as file:  # 打开CSV文件，注意要指定newline=''以避免空行
        writer = csv.writer(file)
        writer.writerows(data)  # 将数据写入CSV文件

def up_sample(BCActions):
    target_indices = np.where(BCActions[:, 3] == 1)[0]
    return target_indices


def main():
    data_dir = "data_collection/data/low_blood/"

    npstate, npactions = read_data(data_dir)
    print(npstate)
    print(npactions)

if __name__ == '__main__':
    main()