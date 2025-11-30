import argparse
import copy
import numpy as np
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import multiprocessing as mp

from pretraining.utils import get_my_trt_model, train_opponent_index, test_opponent_index, create_dir_if_not_exists, set_global_seed, get_env_and_oppo, test, LOG
from testing.search import Searcher


def main(args):
    # * set the seed
    set_global_seed(args.seed, with_tf=False)
    # * get env and opponent policies
    if args.test_mode == "seen":
        args.test_oppo_pi_name = train_opponent_index
    elif args.test_mode == "unseen":
        args.test_oppo_pi_name = test_opponent_index
    elif args.test_mode == "mix":
        args.test_oppo_pi_name = np.concatenate((train_opponent_index, test_opponent_index))
    env_and_oppo = get_env_and_oppo(args)
    with open(f"test_seq/{args.test_mode}_oppo_switch_{args.switch_interval}.npy", 'rb') as f:
        test_oppo_seq = np.load(f)
        args.test_oppo_seq = test_oppo_seq
        
    # * set the result dir
    exp_name = f'{args.env_type}/{args.test_mode}/switch_{args.switch_interval}'
    seed_path = f"{exp_name}/ours-{args.exp_id}/seed{args.seed}/"
    args.seed_result_dir = args.result_dir + seed_path
    create_dir_if_not_exists(args.seed_result_dir)
    args.seed_result_path = args.seed_result_dir + "returns.csv"

    with open(args.seed_result_path, 'w') as f:
        f.write("episode,oppo_pi_idx,return\n")

    # * initialize the model
    if args.load_model_path == "":
        base_dir = "../pretraining/models/"
        args.load_model_path = base_dir + f"{args.env_type}/ours-{args.load_exp_id}/seed{args.seed}/model_iter_{args.load_iter}"
    my_model = get_my_trt_model(args)

    if args.env_type == "oc":
        fake_env = env_and_oppo["env"].copy()
    elif args.env_type == "lbf":
        fake_env = copy.deepcopy(env_and_oppo["env"])
    elif args.env_type == "pp":
        fake_env = copy.deepcopy(env_and_oppo["env"])
    
    searcher = Searcher(
        args_dict=args.__dict__,
        fake_env=fake_env,
        pi_fn=my_model,
    )
    LOG.info("Start testing.")
    test(env_and_oppo, searcher, args)
    LOG.info(f"Finish testing.")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_type", type=str, default="oc", choices=["oc", "lbf", "pp"])
    argparser.add_argument("--device", type=str, default="cuda:0")
    
    argparser.add_argument("--load_model_path", type=str, default="")
    argparser.add_argument("--load_exp_id", type=str, default="v0")
    argparser.add_argument("--load_iter", type=int, default=3999)
    
    argparser.add_argument("--test_mode", type=str, default="seen", choices=["seen", "unseen", "mix"])
    argparser.add_argument("--switch_interval", default=20, choices=[2, 5, 10, 20, "D"])
    argparser.add_argument("--exp_id", type=str, default="v0")
    argparser.add_argument("--seed", type=int)
    
    argparser.add_argument("--num_rollout_per_action", type=int, default=3)
    argparser.add_argument("--rollout_length", type=int, default=3)
    argparser.add_argument("--search_gamma", type=float, default=0.7)
    
    argparser.add_argument("--history_len", type=int, default=20)
    argparser.add_argument("--prompt_len", type=int, default=5)
    argparser.add_argument("--prompt_epi", type=int, default=3)

    argparser.add_argument("--result_dir", type=str, default="results/")
    
    args = argparser.parse_args()
    
    NUM_RUN = 5
    ctx = mp.get_context('spawn')
    subproc = []
    for i in range(NUM_RUN):
        args.seed = i
        p = ctx.Process(target=main, args=(args,))
        p.start()
        subproc.append(p)
    for p in subproc:
        p.join()