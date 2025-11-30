import argparse
import numpy as np
from tqdm import tqdm
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pretraining.config as CONFIG
from pretraining.utils2 import load_dict_from_txt, save_pickle
from pretraining.utils import (
    LOG,
    PBT_DATA_DIR,
    pbt_model_paths,
    AgentGroupObs,
    set_global_seed,
)
from harfang_utils.utils import get_rollouts_harfang
from pretraining.ppo import RLPolicy
from envs.HarfangEnv_GYM.HarfangEnv_GYM import HarfangLowBloodEnvNew


def main(args):
    # * set the seed
    # set_global_seed(args.seed, with_tf=False)
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
    
    save_dir = f"pretraining/models/{env_type}/{data_type}/"
    my_policies = [
        # RLPolicy(env_type, save_dir + f"pi{i}/model.pt", args.device) for i in range(len(test_oppo_policy))
    ]

    # * generate the dataset
    # agent_idxs = my_index_dict[args.env_type]
    if env_type == "Harfang":
        agent_idxs = CONFIG.AGENT_INDEX
        oppo_idxs = CONFIG.OPPO_INDEX

    result_dict = {}
    for i in tqdm(range(len(test_oppo_policy)), desc="oppo_pi"):
        i_name = f"oppo_policy_{test_oppo_policy[i]}"
        result_dict_ = dict()
        for j in tqdm(range(len(agent_idxs)), desc="agent_pi"):
            if env_type == "Harfang":
                agent_group = AgentGroupObs(agent_idxs[j], test_oppo_policy[i], sim_threads=0)
                r_dict = get_rollouts_harfang(env, env_type, agent_group, num_rounds // len(test_oppo_policy), agent_idx=agent_idxs)
            avg_ep_returns = np.mean(np.stack(r_dict['ep_returns'], axis=0), axis=0)
            LOG.info(f"Test type: {data_type}; Opponent: [{test_oppo_policy[i]}]; avg returns result: {avg_ep_returns}")
            
            if result_dict_ == dict():
                result_dict_ = r_dict
            else:
                for k, v in result_dict_.items():
                    result_dict_[k] = np.concatenate((v, r_dict[k]), axis=0)
        result_dict[i_name] = result_dict_
    
    if not os.path.exists(f"pretraining/prompt/"):
        os.makedirs(f"pretraining/prompt/")
    save_pickle(result_dict, f"pretraining/prompt/{env_type}_{data_type}_r{num_rounds}_{exp_id}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_type", type=str, default="Harfang", choices=["oc", "lbf", "pp","Harfang"])
    argparser.add_argument("--data_type", type=str, default="seen", choices=["seen", "unseen", "mixed"])
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--exp_id", type=str, default="v0")
    argparser.add_argument("--num_rounds", type=int, default=1000)
    
    args = argparser.parse_args()
    main(args)