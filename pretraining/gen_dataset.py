import argparse
import time
import gym
import numpy as np
from tqdm import tqdm
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pretraining.utils import (
    LOG,
    PBT_DATA_DIR,
    my_index_dict,
    train_opponent_index,
    pbt_model_paths,
    oc_env_params,
    AgentGroupObs,
    get_rollouts,
    reset_tf,
    set_global_seed,
    get_pbt_agent_from_config,
)

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_pickle, load_pickle, load_dict_from_txt, save_as_json

from pretraining.ppo import RLPolicy

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from multiagent.scenarios.simple_tag import Scenario
from multiagent.environment import MultiAgentEnv

import lbforaging


def main(args):
    # * set the seed
    set_global_seed(args.seed)
    
    # * initialize the envs
    pbt_save_dir = PBT_DATA_DIR[args.env_type] + pbt_model_paths[args.env_type] + "/"
    pbt_config = load_dict_from_txt(pbt_save_dir + "config")
    if args.env_type == "oc":
        mdp = OvercookedGridworld.from_layout_name(**oc_env_params["mdp_params"])
        env = OvercookedEnv(mdp, **oc_env_params["env_params"])
    elif args.env_type == "lbf":
        env = gym.make("Foraging-9x9-2p-5f-v2")
    elif args.env_type == "pp":
        scenario = Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
    
    if args.env_type == "oc":
        data_type = "sp"
    elif args.env_type == "lbf":
        data_type = "br"
    elif args.env_type == "pp":
        data_type = "br"
    
    # * initialize the opponent policies
    opponent_policies = [get_pbt_agent_from_config(pbt_save_dir, pbt_config["sim_threads"], seed=s, agent_idx=idx, best=True, model_type="onnx", device=args.device, env_type=args.env_type) for (s, idx) in train_opponent_index]
    
    if data_type == "br":
        if args.env_type == "lbf":
            save_dir = "models/rl/lbf/BR-v0/"
            my_policies = [RLPolicy(args.env_type, save_dir + f"pi{i}/seed{i}/params_50000.pt", args.device) for i in range(len(train_opponent_index))]
        elif args.env_type == "pp":
            save_dir = "models/rl/pp/BR-v0/"
            my_policies = [RLPolicy(args.env_type, save_dir + f"pi{i}/seed{i}/params_50000.pt", args.device) for i in range(len(train_opponent_index))]
    
    # * generate the dataset
    my_idxs = my_index_dict[args.env_type]
    result_dict = {}
    for i in tqdm(range(len(opponent_policies))):
        i_name = f"s{train_opponent_index[i][0]}_idx{train_opponent_index[i][1]}"
        if args.env_type == "oc":
            agent_group = AgentGroupObs(opponent_policies[i], opponent_policies[i], sim_threads=pbt_config["sim_threads"])
            result_dict_ = get_rollouts(env, args.env_type, agent_group, args.num_rounds // len(train_opponent_index), agent_idx=my_idxs[0])
        elif args.env_type == "lbf":
            agent_group = AgentGroupObs(my_policies[i], opponent_policies[i], sim_threads=pbt_config["sim_threads"])
            result_dict_ = get_rollouts(env, args.env_type, agent_group, args.num_rounds, agent_idx=my_idxs[0])
        elif args.env_type == "pp":
            agent_group = AgentGroupObs(opponent_policies[i], opponent_policies[i], opponent_policies[i], my_policies[i], sim_threads=pbt_config["sim_threads"])
            result_dict_ = get_rollouts(env, args.env_type, agent_group, args.num_rounds, agent_idx=my_idxs[0])
        avg_ep_returns = np.mean(np.stack(result_dict_['ep_returns'], axis=0), axis=0)
        LOG.info(f"Opponent pi [{train_opponent_index[i]}] {data_type} result: {avg_ep_returns}, my_idx: {my_idxs[0]}")
        result_dict[i_name] = result_dict_
    
    if not os.path.exists(f"data/"):
        os.makedirs(f"data/")
    save_pickle(result_dict, f"data/{args.env_type}_{data_type}_r{args.num_rounds}_{args.exp_id}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_type", type=str, default="oc", choices=["oc", "lbf", "pp"])
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--exp_id", type=str, default="v0")
    argparser.add_argument("--num_rounds", type=int, default=1000)
    
    args = argparser.parse_args()
    main(args)