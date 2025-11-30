import argparse
import time
import torch
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pretraining.nets import GPTModel
from pretraining.nn_trainer import Trainer
from pretraining.utils import create_dir_if_not_exists, get_prompt, get_prompt_batch, load_dataset, set_global_seed, CrossEntropy, MeanSquareError, LOG
import multiprocessing as mp


def main(args):
    # * set the seed
    set_global_seed(args.seed, with_tf=False)
    # * set the result and model dir
    group_name = f'{args.env_type}/ours-{args.exp_id}'
    exp_name = f'{group_name}/seed{args.seed}'

    args.seed_model_dir = args.model_dir + f"{exp_name}/"
    create_dir_if_not_exists(args.seed_model_dir)

    args.seed_result_dir = args.result_dir + f"{exp_name}/"
    create_dir_if_not_exists(args.seed_result_dir)

    args.seed_loss_path = args.seed_result_dir + "loss.csv"
    with open(args.seed_loss_path, 'w') as f:
        f.write("train_step,action_loss,value_loss,oppo_pi_loss\n")

    # * load the dataset
    dataset = load_dataset(args)
    prompt_dataset = load_dataset(args, prompt=True)
    LOG.info("Finish loading dataset.")
    
    get_prompt_batch_fn = get_prompt_batch(dataset, prompt_dataset, args)
    get_prompt_fn = get_prompt(prompt_dataset, args)
    # * initialize the model
    model = GPTModel(args=args)
    model = model.to(device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1))
    # * initialize the trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        get_prompt_batch_fn=get_prompt_batch_fn,
        get_prompt_fn=get_prompt_fn,
        act_loss_fn=CrossEntropy,
        value_loss_fn=MeanSquareError,
        args=args,
    )
    # * train and save the model
    LOG.info("Start pretraining.")
    for i in range(args.num_iter):
        _, action_loss, value_loss, oppo_pi_loss = trainer.train(num_update=args.num_update)
        LOG.info(f"iter [{i}]: finish training")
        with open(args.seed_loss_path, 'a') as f:
            f.write(f"{i+1},{action_loss:.6f},{value_loss:.6f},{oppo_pi_loss:.6f}\n")
        if i % args.ckpt_freq == 0 or i == args.num_iter-1:
            model_path = args.seed_model_dir+f"model_iter_{i}"
            trainer.save_model(model_path)
            # trainer.save_onnx_model(model_path)
            LOG.info(f"iter [{i}]: finish * saving model *")
    LOG.info(f"Finish pretraining.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_type", type=str, default="oc", choices=["oc", "lbf", "pp"])
    argparser.add_argument("--exp_id", type=str, default="v0")
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--load_data_path", type=str, default="data/oc_sp_r1000_v0")
    # argparser.add_argument("--load_data_path", type=str, default="data/lbf_br_r1000_v0")
    # argparser.add_argument("--load_data_path", type=str, default="data/st_br_r1000_v0")
    argparser.add_argument("--seed", type=int)
    
    argparser.add_argument("--num_iter", type=int, default=4000)
    argparser.add_argument("--ckpt_freq", type=int, default=80)
    argparser.add_argument("--num_update", type=int, default=10)
    argparser.add_argument("--vf_coef", type=float, default=0.5)
    argparser.add_argument("--oppo_pi_coef", type=float, default=0.8)
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--learning_rate", type=float, default=6e-4)
    argparser.add_argument("--weight_decay", type=float, default=1e-4)
    argparser.add_argument("--warmup_steps", type=int, default=1e4)
    
    argparser.add_argument("--history_len", type=int, default=20)
    argparser.add_argument("--prompt_len", type=int, default=5)
    argparser.add_argument("--prompt_epi", type=int, default=3)
    argparser.add_argument("--hidden_dim", type=int, default=32)
    argparser.add_argument("--n_inner", type=int, default=4 * 32)
    argparser.add_argument("--activation_function", type=str, default="gelu_new")
    argparser.add_argument("--n_layer", type=int, default=3)
    argparser.add_argument("--n_head", type=int, default=1)
    argparser.add_argument("--n_positions", type=int, default=512)
    argparser.add_argument("--resid_pdrop", type=float, default=0.1)
    argparser.add_argument("--attn_pdrop", type=float, default=0.1)
    argparser.add_argument("--clip_grad", type=float, default=0.5)
    argparser.add_argument("--action_tanh", type=bool, default=False)
    
    argparser.add_argument("--result_dir", type=str, default="results/sl/")
    argparser.add_argument("--model_dir", type=str, default="models/sl/")
    
    args = argparser.parse_args()
    
    main(args)