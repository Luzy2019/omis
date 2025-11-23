import argparse
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import random
import numpy as np
import gym
from collections import namedtuple
import logging
import torch
from pretraining.ppo import PPO
from pretraining.utils import PBT_DATA_DIR, horizon_per_ep_dict, my_index_dict, opponent_index_dict, my_obs_dim_dict, act_dim_dict, train_opponent_index, pbt_model_paths, oc_env_params, get_pbt_agent_from_config

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.utils import load_dict_from_txt

import lbforaging

from multiagent.scenarios.simple_tag import Scenario
from multiagent.environment import MultiAgentEnv
import multiprocessing as mp


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def get_oppo_policies(env_type, oppo_policy_idx, oppo_idxs, device):
    all_policies = {idx: None for idx in oppo_idxs}
    pbt_save_dir = PBT_DATA_DIR[env_type] + pbt_model_paths[env_type] + "/"
    pbt_config = load_dict_from_txt(pbt_save_dir + "config")
    for i in oppo_idxs:
        s, idx = train_opponent_index[oppo_policy_idx]
        p = get_pbt_agent_from_config(pbt_save_dir, pbt_config["sim_threads"], seed=s, agent_idx=idx, best=True, model_type="onnx", device=device, env_type=env_type)
        all_policies[i] = p
    return all_policies


def main(args):
    Transition = namedtuple('Transition', ['state', 'action', 'a_prob', 'reward', 'next_state'])

    exp_id = args.exp_id
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_type = args.env_type
    num_steps = horizon_per_ep_dict[env_type]
    num_episodes = args.num_episodes
    gamma = args.gamma
    
    device = args.device
    checkpoint_freq = args.checkpoint_freq
    batch_size = args.batch_size
    num_update_per_iter = args.num_update_per_iter
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    hidden_dim = args.hidden_dim
    clip_param = args.clip_param
    max_grad_norm = args.max_grad_norm
    
    my_idxs = my_index_dict[env_type]
    MID = my_idxs[0]
    oppo_idxs = opponent_index_dict[env_type]
    oppo_policy_idx = args.oppo_policy_idx

    save_dir = args.save_dir + f"{env_type}/BR-{exp_id}/pi{oppo_policy_idx}/seed{seed}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    if env_type == "oc":
        # oppo_idxs = opponent_index_dict[env_type]
        mdp = OvercookedGridworld.from_layout_name(**oc_env_params["mdp_params"])
        env = OvercookedEnv(mdp, **oc_env_params["env_params"])
    elif env_type == "lbf":
        env = gym.make("Foraging-9x9-2p-5f-v2")
    elif env_type == "pp":
        # load scenario from script
        scenario = Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
    
    state_dim = my_obs_dim_dict[env_type]
    action_dim = act_dim_dict[env_type]

    agent = PPO(state_dim, hidden_dim, action_dim, device,
                num_steps, batch_size, actor_lr, critic_lr,
                gamma, num_update_per_iter, clip_param, max_grad_norm)

    oppo_return_list = []
    return_list = []
    
    oppo_policies = get_oppo_policies(env_type, oppo_policy_idx, oppo_idxs, device)
    
    for i in range(num_episodes):
        ep_return = [0 for _ in range(len(oppo_idxs+my_idxs))]
        if env_type == "oc":
            env.reset()
            obs_n = env.mdp.lossless_state_encoding(env.state)
        elif env_type in ["lbf", "pp"]:
            obs_n = env.reset()
        if env_type == "pp":
            t = 0
            done = False
        while True:
            act_n = [None for _ in range(len(oppo_idxs+my_idxs))]
            act_index_n = [None for _ in range(len(oppo_idxs+my_idxs))]
            for j in oppo_idxs:
                oppo_act_idx = oppo_policies[j](obs_n[j])
                if env_type == "oc":
                    act_n[j] = Action.INDEX_TO_ACTION[oppo_act_idx]
                elif env_type == "lbf":
                    act_n[j] = oppo_act_idx
                elif env_type == "pp":
                    act_n[j] = np.eye(action_dim)[oppo_act_idx]
                act_index_n[j] = oppo_act_idx

            _, act_index, act_prob = agent.select_action(obs_n[MID])
            if env_type == "oc":
                act_n[MID] = Action.INDEX_TO_ACTION[act_index]
            elif env_type == "lbf":
                act_n[MID] = act_index
            elif env_type == "pp":
                act_n[MID] = np.eye(action_dim)[act_index]
            act_index_n[MID] = act_index
            
            if env_type == "oc":
                next_state, reward, done, info = env.step(act_n)
                reward_n = [reward for _ in range(len(oppo_idxs+my_idxs))]
                next_obs_n = env.mdp.lossless_state_encoding(next_state)
            elif env_type == "lbf":
                next_obs_n, reward_n, done_n, info = env.step(act_n)
                done = any(done_n)
            elif env_type == "pp":
                next_obs_n, reward_n, _, info = env.step(act_n)
                t += 1
                if t >= num_steps:
                    done = True

            trans = Transition(obs_n[MID], act_index_n[MID], act_prob, reward_n[MID], next_obs_n[MID])
            agent.store_transition(trans)

            for j in oppo_idxs:
                ep_return[j] += reward_n[j]
            
            ep_return[MID] += reward_n[MID]
            obs_n = next_obs_n
            
            if done:
                break

        avg_oppo_ep_return = np.mean([ep_return[j] for j in oppo_idxs])
        oppo_return_list.append(avg_oppo_ep_return)
        
        return_list.append(ep_return[MID])
        
        if len(agent.buffer) >= agent.batch_size:
            agent.update()
        
        cur_ep = i + 1
        
        if cur_ep % checkpoint_freq == 0:
            path = save_dir + f'/params_{cur_ep}'
            agent.save_params(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, default='oc', choices=['oc', 'lbf', 'pp'])
    parser.add_argument("--exp_id", type=str, default='v0')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--oppo_policy_idx", type=int)
    parser.add_argument("--seed", type=int)
    
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--checkpoint_freq", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_update_per_iter", type=int, default=10)
    parser.add_argument("--actor_lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    
    parser.add_argument("--save_dir", type=str, default='models/rl/')
    
    args = parser.parse_args()
    
    NUM_RUN = len(train_opponent_index)
    ctx = mp.get_context('spawn')
    subproc = []
    for i in range(NUM_RUN):
        args.oppo_policy_idx = i
        args.seed = i
        p = ctx.Process(target=main, args=(args,))
        p.start()
        subproc.append(p)
    for p in subproc:
        p.join()