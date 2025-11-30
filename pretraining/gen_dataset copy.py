
"""
env modules:
- overcooked_ai_py
- multiagent_particle_envs
- lb_foraging_envs
- Harfang
"""

import argparse
import time
import numpy as np
from tqdm import tqdm
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pretraining.config as CONFIG
from pretraining.ppo import RLPolicy
from pretraining.utils2 import load_dict_from_txt, save_pickle
from pretraining.utils import (
    LOG,
    PBT_DATA_DIR,
    pbt_model_paths,
    AgentGroupObs,
    get_rollouts,
    set_global_seed,
)
from envs.HarfangEnv_GYM.HarfangEnv_GYM import HarfangEnv


def main(args):
    set_global_seed(args.seed)
    pbt_save_dir = PBT_DATA_DIR[args.env_type] + pbt_model_paths[args.env_type] + "/"
    pbt_config = load_dict_from_txt(pbt_save_dir + "config")
    
    seen_oppo_policies = CONFIG.SEEN_OPPO_POLICY
    unseen_oppo_policies = CONFIG.UNSEEN_OPPO_POLICY

    if args.data_type == "seen":
        test_oppo_policy = seen_oppo_policies
    elif args.data_type == "unseen":
        test_oppo_policy = unseen_oppo_policies
    elif args.data_type == "mixed":
        test_oppo_policy = np.concatenate([seen_oppo_policies, unseen_oppo_policies])

    if args.env_type == "Harfang":
        env = HarfangEnv()
    
    save_dir = f"models/{args.env_type}/"
    my_policies = [RLPolicy(args.env_type, save_dir + f"pi{i}/seed{i}/params_50000.pt", args.device) for i in range(len(test_oppo_policy))]

    # * generate the dataset
    # my_idxs = my_index_dict[args.env_type]
    if args.env_type == "Harfang":
        my_idxs = CONFIG.AGENT_INDEX

    result_dict = {}
    for i in tqdm(range(len(test_oppo_policy))):
        i_name = f"s{test_oppo_policy[i]}_idx{CONFIG.OPPO_INDEX[0]}"
        if args.env_type == "Harfang":
            agent_group = AgentGroupObs(my_policies[0], test_oppo_policy[i], sim_threads=pbt_config["sim_threads"])
            result_dict_ = get_rollouts(env, args.env_type, agent_group, args.num_rounds // len(test_oppo_policy), agent_idx=my_idxs[0])
        avg_ep_returns = np.mean(np.stack(result_dict_['ep_returns'], axis=0), axis=0)
        LOG.info(f"Opponent pi [{test_oppo_policy[i]}] {args.data_type} result: {avg_ep_returns}, my_idx: {my_idxs[0]}")
        result_dict[i_name] = result_dict_
    
    if not os.path.exists(f"data/"):
        os.makedirs(f"data/")
    save_pickle(result_dict, f"data/{args.env_type}_{args.data_type}_r{args.num_rounds}_{args.exp_id}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_type", type=str, default="Harfang", choices=["oc", "lbf", "pp", "Harfang"])
    argparser.add_argument("--data_type", type=str, default="seen", choices=["seen", "unseen", "mixed"])
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--exp_id", type=str, default="v0")
    argparser.add_argument("--num_rounds", type=int, default=100)
    args = argparser.parse_args()

    args.exp_id = time.strftime("%Y%m%d%H%M%S", time.localtime())

    main(args)