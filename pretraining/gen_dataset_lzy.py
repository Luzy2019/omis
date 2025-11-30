import argparse
import numpy as np
from tqdm import tqdm
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import time
import pretraining.config as CONFIG
from pretraining.utils2 import save_pickle
from pretraining.utils import (
    LOG,
    AgentGroupObs,
)
from harfang_utils.utils import get_rollouts_harfang


def main(args):
    # * set the seed
    # set_global_seed(args.seed)
    # * initialize the envs
    # pbt_save_dir = PBT_DATA_DIR[args.env_type] + pbt_model_paths[args.env_type] + "/"
    # pbt_config = load_dict_from_txt(pbt_save_dir + "config")
    
    seen_oppo_policies = CONFIG.SEEN_OPPO_POLICY
    unseen_oppo_policies = CONFIG.UNSEEN_OPPO_POLICY

    data_type = args.data_type
    env_type = args.env_type
    num_rounds = args.num_rounds
    exp_id = args.exp_id

    if data_type == "seen":
        test_oppo_policy = seen_oppo_policies
    elif data_type == "unseen":
        test_oppo_policy = unseen_oppo_policies
    elif data_type == "mixed":
        test_oppo_policy = np.concatenate([seen_oppo_policies, unseen_oppo_policies])

    if env_type == "Harfang":
        env = None
    
    # save_dir = f"models/{env_type}/"
    # my_policies = [
    #     RLPolicy(env_type, save_dir + f"pi{i}/model.pt", args.device) for i in range(len(test_oppo_policy))
    # ]

    # * generate the dataset
    # agent_idxs = my_index_dict[env_type]
    if env_type == "Harfang":
        agent_idxs = CONFIG.AGENT_INDEX
        oppo_idxs = CONFIG.OPPO_INDEX

    result_dict = {}
    for i in tqdm(range(len(test_oppo_policy)), desc="oppo_pi"):
        i_name = f"oppo_policy_{test_oppo_policy[i]}"
        if env_type == "Harfang":
            agent_group = AgentGroupObs(agent_idxs[i], test_oppo_policy[i], sim_threads=0)
            r_dict = get_rollouts_harfang(env, env_type, agent_group, num_rounds // len(test_oppo_policy), agent_idx=agent_idxs)
        avg_ep_returns = np.mean(np.stack(r_dict['ep_returns'], axis=0), axis=0)
        
        LOG.info(f"Test type: {data_type}; Opponent: [{test_oppo_policy[i]}]; avg returns result: {avg_ep_returns}")
        result_dict[i_name] = r_dict
    
    if not os.path.exists(f"pretraining/data/"):
        os.makedirs(f"pretraining/data/")
    save_pickle(result_dict, f"pretraining/data/{env_type}_{data_type}_r{num_rounds}_{exp_id}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_type", type=str, default="Harfang", choices=["oc", "lbf", "pp", "Harfang"])
    argparser.add_argument("--data_type", type=str, default="seen", choices=["seen", "unseen", "mixed"])
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--exp_id", type=str, default="v2")
    argparser.add_argument("--num_rounds", type=int, default=100)
    args = argparser.parse_args()

    # args.exp_id = time.strftime("%Y%m%d%H%M%S", time.localtime())

    main(args)