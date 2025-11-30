import logging
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import re
# import gym
import copy
# import onnx
# import onnxruntime as ort
import tqdm
import random
import numpy as np
# from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F
from pretraining.utils2 import load_pickle

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
LOG = logging.getLogger()

# * ============================== hyper-parameters ============================== * #
opponent_population_agent_index = [0,1,2,3,4]
opponent_population_seed = [
    9015,7267,
    286,4756,80,
]

train_opponent_index = [
    (s, idx) for idx in opponent_population_agent_index for s in opponent_population_seed[:2]
]
test_opponent_index = [
    (s, idx) for idx in opponent_population_agent_index for s in opponent_population_seed[2:4]
]

print(train_opponent_index)

my_index_dict = {
    "oc": [0],
    "lbf": [0],
    "pp": [3],
    "Harfang": [0],
}

opponent_index_dict = {
    "oc": [1],
    "lbf": [1],
    "pp": [0,1,2],
    "Harfang": [1],
}

my_obs_dim_dict = {
    "oc": (5, 4, 20),
    "lbf": (21,),
    "pp": (14,),
    "Harfang": (13,),
}

oppo_obs_dim_dict = {
    "oc": (5, 4, 20),
    "lbf": (21,),
    "pp": (16,),
    "Harfang": (6,),
}

act_dim_dict = {
    "oc": 6,
    "lbf": 6,
    "pp": 5,
    "Harfang": 4,
}

horizon_per_ep_dict = {
    "oc": 400,
    "lbf": 50,
    "pp": 100,
    "Harfang": 4000,
}

reward_scale_dict = {
    "oc": 100.,
    "lbf": 1.,
    "pp": 100.,
    "Harfang": 100.,
}

PBT_DATA_DIR = {
    "oc": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../envs/overcooked_ai_envs/models/"),
    "lbf": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../envs/lb_foraging_envs/models/"),
    "pp": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../envs/multiagent_particle_envs/models/"),
    "Harfang": None,
}

pbt_model_paths = {
    "oc": "pbt_simple",
    "lbf": "pbt_lbf",
    "pp": "pbt_st",
    "Harfang": None,
}

oc_env_params = {
    "mdp_params": {
        "layout_name": "simple",
        "start_order_list": None,
        "rew_shaping_params": {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0.015,
            "POT_DISTANCE_REW": 0.03,
            "SOUP_DISTANCE_REW": 0.1,
        }
    },
    "env_params": {
        "horizon": 400
    },
}

# * ============================== common utils ============================== * #
def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_global_seed(seed, with_tf=True):
    random.seed(seed)
    np.random.seed(seed)
    if with_tf:
        import tensorflow as tf
        tf.random.set_random_seed(seed)
    torch.manual_seed(seed)

def reset_tf():
    import tensorflow as tf
    """Clean up tensorflow graph and session.
    NOTE: this also resets the tensorflow seed"""
    tf.reset_default_graph()
    if tf.get_default_session() is not None:
        tf.get_default_session().close()


# * ============================== testing utils ============================== * #

def get_env_and_oppo(args):
    if args.env_type == "oc":
        mdp = OvercookedGridworld.from_layout_name(**oc_env_params["mdp_params"])
        env = OvercookedEnv(mdp, **oc_env_params["env_params"])
    elif args.env_type == "lbf":
        env = gym.make("Foraging-9x9-2p-5f-v2")
    elif args.env_type == "pp":
        scenario = Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
    pbt_save_dir = PBT_DATA_DIR[args.env_type] + pbt_model_paths[args.env_type] + "/"
    pbt_config = load_dict_from_txt(pbt_save_dir + "config")
    opponent_policies = [get_pbt_agent_from_config(pbt_save_dir, pbt_config["sim_threads"], seed=s, agent_idx=idx, best=True, model_type="trt", device=args.device, env_type=args.env_type) for s, idx in args.test_oppo_pi_name]
    return dict(
        env=env,
        test_oppo_pi=opponent_policies,
    )

def get_my_trt_model(args):
    import onnx_tensorrt.backend as backend
    model_path = args.load_model_path + ".onnx"
    model = onnx.load(model_path)
    engine = backend.prepare(model, device=args.device.upper())
    batch_size = 1
    K = args.history_len
    PK = args.prompt_len*args.prompt_epi
    my_obs_dim = my_obs_dim_dict[args.env_type]
    act_dim = act_dim_dict[args.env_type]
    oppo_obs_dim = oppo_obs_dim_dict[args.env_type]
    oppo_idxs = opponent_index_dict[args.env_type]
    obs = np.random.randn(batch_size, K, *my_obs_dim).astype(np.float32, order='C')
    actions = np.random.randn(batch_size, K, act_dim).astype(np.float32, order='C')
    oppo_actions = []
    for _ in range(len(oppo_idxs)):
        oppo_actions.append(np.random.randn(batch_size, K, act_dim).astype(np.float32))
    oppo_actions = np.stack(oppo_actions, axis=0).astype(np.float32, order='C')
    returns_to_go = np.random.randn(batch_size, K, 1).astype(np.float32, order='C')
    timesteps = np.ones((batch_size, K), dtype=np.int32, order='C')
    attention_mask = np.ones((batch_size, K), dtype=np.int32, order='C')
    prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask = [], [], [], []
    for _ in range(len(oppo_idxs)):
        prompt_states.append(np.random.randn(batch_size, PK, *oppo_obs_dim).astype(np.float32, order='C'))
        prompt_actions.append(np.random.randn(batch_size, PK, act_dim).astype(np.float32, order='C'))
        prompt_timesteps.append(np.ones((batch_size, PK), dtype=np.int32, order='C'))
        prompt_attention_mask.append(np.ones((batch_size, PK), dtype=np.int32, order='C'))
    prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask = \
        np.stack(prompt_states, axis=0).astype(np.float32, order='C'), np.stack(prompt_actions, axis=0).astype(np.float32, order='C'), np.stack(prompt_timesteps, axis=0).astype(np.int32, order='C'), np.stack(prompt_attention_mask, axis=0).astype(np.int32, order='C')
    input_data_ = obs, actions, oppo_actions, returns_to_go, timesteps, attention_mask, prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask
    engine.run(input_data_)
    
    def step_fn(obs, actions, oppo_actions, returns_to_go, timesteps, attention_mask, prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask):
        input_data = {
            'obs': obs,
            'actions': actions,
            'oppo_actions': oppo_actions,
            'returns_to_go': returns_to_go,
            'timesteps': timesteps,
            'attention_mask': attention_mask,
            'prompt_states': prompt_states,
            'prompt_actions': prompt_actions,
            'prompt_timesteps': prompt_timesteps,
            'prompt_attention_mask': prompt_attention_mask,
        }
        output_data = engine.engine.run(input_data)
        value, act, oppo_act = output_data
        # act_dim,
        action_preds = act[0, -1]
        # 1,
        value_preds = value[0, -1]
        # num_oppo, act_dim
        oppo_action_preds = oppo_act[:, 0, -1]
        return action_preds, value_preds, oppo_action_preds
    return step_fn


def sample_prompt(oppo_traj_window, prompt_len, prompt_epi, oppo_idxs, oppo_obs_dim, act_dim, num_steps):
    num_oppo = len(oppo_idxs)
    o_oppo, a_oppo, timestep_oppo, mask_oppo = [], [], [], []
    if oppo_traj_window != None:
        num_padding = prompt_epi - len(oppo_traj_window)
        for _ in range(num_padding):
            o_oppo.append(np.zeros((num_oppo, 1, prompt_len, *oppo_obs_dim), dtype=np.float32))
            a_oppo.append(np.ones((num_oppo, 1, prompt_len, act_dim), dtype=np.float32) * -10.)
            timestep_oppo.append(np.zeros((num_oppo, 1, prompt_len), dtype=np.int64))
            mask_oppo.append(np.zeros((num_oppo, 1, prompt_len), dtype=np.int64))
        for oppo_traj in oppo_traj_window:
            o_oppo_wd, a_oppo_wd, timestep_oppo_wd, mask_oppo_wd = [], [], [], []
            for i in range(num_oppo):
                o_oppo_, a_oppo_, timestep_oppo_ = oppo_traj[i]
                si = np.random.randint(0, o_oppo_.shape[0])
                o_oppo_wd.append(o_oppo_[si:si + prompt_len, :].reshape(1, 1, -1, *oppo_obs_dim))
                a_oppo_wd.append(a_oppo_[si:si + prompt_len, :].reshape(1, 1, -1, act_dim))
                timestep_oppo_wd.append(timestep_oppo_[:, si:si + prompt_len].reshape(1, 1, -1))
                timestep_oppo_wd[-1][timestep_oppo_wd[-1] >= num_steps] = num_steps - 1  # padding cutoff
                
                tlen = o_oppo_wd[-1].shape[2]
                
                o_oppo_wd[-1] = np.concatenate([np.zeros((1, 1, prompt_len - tlen, *oppo_obs_dim), dtype=np.float32), o_oppo_wd[-1]], axis=2)
                a_oppo_wd[-1] = np.concatenate([np.ones((1, 1, prompt_len - tlen, act_dim), dtype=np.float32) * -10., a_oppo_wd[-1]], axis=2)
                timestep_oppo_wd[-1] = np.concatenate([np.zeros((1, 1, prompt_len - tlen), dtype=np.int64), timestep_oppo_wd[-1]], axis=2)
                mask_oppo_wd.append(np.concatenate([np.zeros((1, 1, prompt_len - tlen), dtype=np.int64), np.ones((1, 1, tlen), dtype=np.int64)], axis=2, dtype=np.int64))
            o_oppo.append(np.concatenate(o_oppo_wd, axis=0, dtype=np.float32))
            a_oppo.append(np.concatenate(a_oppo_wd, axis=0, dtype=np.float32))
            timestep_oppo.append(np.concatenate(timestep_oppo_wd, axis=0, dtype=np.int64))
            mask_oppo.append(np.concatenate(mask_oppo_wd, axis=0, dtype=np.int64))
    else:
        num_padding = prompt_epi
        for _ in range(num_padding):
            o_oppo.append(np.zeros((num_oppo, 1, prompt_len, *oppo_obs_dim), dtype=np.float32))
            a_oppo.append(np.ones((num_oppo, 1, prompt_len, act_dim), dtype=np.float32) * -10.)
            timestep_oppo.append(np.zeros((num_oppo, 1, prompt_len), dtype=np.int64))
            mask_oppo.append(np.zeros((num_oppo, 1, prompt_len), dtype=np.int64))
    
    o_oppo = np.concatenate(o_oppo, axis=2).astype(np.float32, order='C')
    a_oppo = np.concatenate(a_oppo, axis=2).astype(np.float32, order='C')
    timestep_oppo = np.concatenate(timestep_oppo, axis=2).astype(np.int64, order='C')
    mask_oppo = np.concatenate(mask_oppo, axis=2).astype(np.int64, order='C')
    
    return o_oppo, a_oppo, timestep_oppo, mask_oppo


def prepare_my_input(obs, actions, oppo_actions, returns_to_go, timesteps, my_obs_dim, act_dim, oppo_idxs, history_len,):
    obs = obs.reshape(1, -1, *my_obs_dim)
    actions = actions.reshape(1, -1, act_dim)
    
    argsorted_oppo_idxs = np.argsort(oppo_idxs)
    oppo_actions_ = []
    for i in argsorted_oppo_idxs:
        oppo_actions_.append(oppo_actions[oppo_idxs[i]].reshape(1, 1, -1, act_dim))
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    obs = obs[:, -history_len:]
    actions = actions[:, -history_len:]
    for i in range(len(oppo_idxs)):
        oppo_actions_[i] = oppo_actions_[i][:, :, -history_len:]
    returns_to_go = returns_to_go[:, -history_len:]
    timesteps = timesteps[:, -history_len:]
    
    cur_seq_length = obs.shape[1]
    
    if cur_seq_length < history_len:
        obs = np.concatenate(
            [np.zeros((1, history_len-cur_seq_length, *my_obs_dim)), obs],
            axis=1
        )
        actions = np.concatenate(
            [np.zeros((1, history_len - cur_seq_length, act_dim)), actions],
            axis=1
        )
        for i in range(len(oppo_idxs)):
            oppo_actions_[i] = np.concatenate(
                [np.zeros((1, 1, history_len - cur_seq_length, act_dim)), oppo_actions_[i]],
                axis=2
            )
        returns_to_go = np.concatenate(
            [np.zeros((1, history_len-cur_seq_length, 1)), returns_to_go],
            axis=1
        )
        timesteps = np.concatenate(
            [np.zeros((1, history_len-cur_seq_length)), timesteps],
            axis=1
        )
    obs = obs.astype(dtype=np.float32, order='C')
    actions = actions.astype(dtype=np.float32, order='C')
    for i in range(len(oppo_idxs)):
        oppo_actions_[i] = oppo_actions_[i].astype(dtype=np.float32)
    oppo_actions = np.concatenate(oppo_actions_, axis=0).astype(dtype=np.float32, order='C')
    returns_to_go = returns_to_go.astype(dtype=np.float32, order='C')
    timesteps = timesteps.astype(dtype=np.int64, order='C')
    attention_mask = np.concatenate([np.zeros((1, history_len-cur_seq_length), dtype=np.int64), np.ones((1, cur_seq_length), dtype=np.int64)], axis=1).astype(dtype=np.int64, order='C')
    
    return obs, actions, oppo_actions, returns_to_go, timesteps, attention_mask


def eval_a_epi(
    env,
    env_type,
    my_obs_dim,
    oppo_obs_dim,
    act_dim,
    searcher,
    oppo_pi,
    my_idxs,
    oppo_idxs,
    num_steps,
    oppo_traj_window,
    prompt_len,
    prompt_epi,
    history_len,
):  
    argsorted_oppo_idxs = np.argsort(oppo_idxs)
    
    if env_type == "oc":
        env.reset()
        obs_n = env.mdp.lossless_state_encoding(env.state)
    elif env_type in ["lbf", "pp"]:
        obs_n = env.reset()
    
    o_oppo, a_oppo, timestep_oppo, mask_oppo = sample_prompt(oppo_traj_window, prompt_len, prompt_epi, oppo_idxs, oppo_obs_dim, act_dim, num_steps)
    
    obs_n_np = {idx:obs_n[idx].reshape(1, *my_obs_dim if idx in my_idxs else oppo_obs_dim).astype(np.float32, order='C') for idx in oppo_idxs + my_idxs}
    act_n_np = {idx:np.zeros((0, act_dim), dtype=np.float32) for idx in oppo_idxs + my_idxs}
    my_value_np = np.zeros((1, 0), dtype=np.float32)
    timestep_np = np.array(0, dtype=np.int64, order='C').reshape(1, 1)
    
    episode_return = [0. for _ in oppo_idxs + my_idxs]
    true_step = 0
    for t in range(num_steps):
        act_n = [None for _ in oppo_idxs + my_idxs]
        for idx in oppo_idxs + my_idxs:
            act_n_np[idx] = np.concatenate([act_n_np[idx], np.zeros((1, act_dim))], axis=0)
        my_value_np = np.concatenate([my_value_np, np.zeros((1, 1))], axis=1)
        obs, actions, oppo_actions, returns_to_go, timesteps, attention_mask = prepare_my_input(
            obs=obs_n_np[my_idxs[0]],
            actions=act_n_np[my_idxs[0]],
            oppo_actions={idx:act_n_np[idx] for idx in oppo_idxs}, 
            returns_to_go=my_value_np,
            timesteps=timestep_np,
            my_obs_dim=my_obs_dim,
            act_dim=act_dim,
            oppo_idxs=oppo_idxs,
            history_len=history_len,
        )
        cur_model_info = (copy.deepcopy(obs), copy.deepcopy(actions), copy.deepcopy(oppo_actions), copy.deepcopy(returns_to_go), copy.deepcopy(timesteps), copy.deepcopy(attention_mask), copy.deepcopy(o_oppo), copy.deepcopy(a_oppo), copy.deepcopy(timestep_oppo), copy.deepcopy(mask_oppo))
        if env_type == "oc":
            statedict = {
                "state": env.state.deepcopy(),
                "cumulative_sparse_rewards": copy.deepcopy(env.cumulative_sparse_rewards),
                "cumulative_shaped_rewards": copy.deepcopy(env.cumulative_shaped_rewards),
                "t": copy.deepcopy(env.t),
            }
        elif env_type in ["lbf", "pp"]:
            statedict = {
                "root_env": copy.deepcopy(env),
                "t": t,
            }
        my_act_idx, value_preds = searcher.search(statedict, cur_model_info)
        my_action_onehot = np.eye(act_dim, dtype=np.float32, order='C')[my_act_idx]
        if env_type == "oc":
            act_n[my_idxs[0]] = Action.INDEX_TO_ACTION[my_act_idx]
        elif env_type == "lbf":
            act_n[my_idxs[0]] = my_act_idx
        elif env_type == "pp":
            act_n[my_idxs[0]] = my_action_onehot
        act_n_np[my_idxs[0]][-1] = my_action_onehot
        for idx in oppo_idxs:
            oppo_act_idx = oppo_pi(obs_n[idx])
            oppo_action_onehot = np.eye(act_dim, dtype=np.float32, order='C')[oppo_act_idx]
            if env_type == "oc":
                act_n[idx] = Action.INDEX_TO_ACTION[oppo_act_idx]
            elif env_type == "lbf":
                act_n[idx] = oppo_act_idx
            elif env_type == "pp":
                act_n[idx] = oppo_action_onehot
            act_n_np[idx][-1] = oppo_action_onehot
        next_state, reward_n, _, _ = env.step(act_n)
        if env_type == "oc":
            reward = reward_n
            obs_n = env.mdp.lossless_state_encoding(next_state)
        elif env_type in ["lbf", "pp"]:
            reward = reward_n[my_idxs[0]]
            obs_n = next_state
        for idx in oppo_idxs + my_idxs:
            obs_new = obs_n[idx].reshape(1, *my_obs_dim if idx in my_idxs else oppo_obs_dim).astype(dtype=np.float32, order='C')
            obs_n_np[idx] = np.concatenate([obs_n_np[idx], obs_new], axis=0)
        timestep_np = np.concatenate([timestep_np, np.ones((1, 1), dtype=np.int64, order='C') * (t+1)], axis=1)
        my_value_np[0, -1] = value_preds
        for i in oppo_idxs + my_idxs:
            episode_return[i] += reward
        true_step += 1
    
    min_step = min(num_steps, true_step)
    oppo_traj_new = [(
        obs_n_np[oppo_idxs[i]][:min_step, :],
        act_n_np[oppo_idxs[i]][:min_step, :],
        timestep_np[:, :min_step],
    ) for i in argsorted_oppo_idxs]
    if oppo_traj_window == None:
        oppo_traj_window = [oppo_traj_new]
    else:
        oppo_traj_window.append(oppo_traj_new)
    my_epi_ret = episode_return[my_idxs[0]]
    oppo_epi_ret = np.mean([episode_return[i] for i in oppo_idxs])
    
    return my_epi_ret, oppo_epi_ret, oppo_traj_window


def test(env_and_oppo, searcher, args):
    LOG.info(f'Testing against opponents: {args.test_oppo_pi_name}')
    env, test_oppo_pi = env_and_oppo["env"], env_and_oppo["test_oppo_pi"]
    env_type = args.env_type
    my_obs_dim, oppo_obs_dim, act_dim = my_obs_dim_dict[args.env_type], oppo_obs_dim_dict[args.env_type], act_dim_dict[args.env_type]
    my_idxs, oppo_idxs = my_index_dict[args.env_type], opponent_index_dict[args.env_type]
    num_steps = horizon_per_ep_dict[args.env_type]
    prompt_len, history_len = args.prompt_len, args.history_len
    prompt_epi = args.prompt_epi
    test_oppo_seq = args.test_oppo_seq
    if args.switch_interval == "D":
        switch_interval = 1
    else:
        switch_interval = args.switch_interval
    
    returns = []
    oppo_traj_window = None
    for cur_test_oppo_num in range(len(test_oppo_seq)):
        oppo_pi_idx = test_oppo_seq[cur_test_oppo_num]
        oppo_pi = test_oppo_pi[oppo_pi_idx]
        for cur_test_ep in range(switch_interval):
            with torch.no_grad():
                my_epi_ret, _, oppo_traj_window_new = eval_a_epi(
                    env,
                    env_type,
                    my_obs_dim,
                    oppo_obs_dim,
                    act_dim,
                    searcher,
                    oppo_pi,
                    my_idxs,
                    oppo_idxs,
                    num_steps,
                    oppo_traj_window,
                    prompt_len,
                    prompt_epi,
                    history_len,
                )
            oppo_traj_window = oppo_traj_window_new
            oppo_traj_window = oppo_traj_window[-prompt_epi:]
            returns.append(my_epi_ret)
            with open(args.seed_result_path, 'a') as f:
                f.write(f"{cur_test_ep+cur_test_oppo_num*switch_interval},{oppo_pi_idx},{my_epi_ret}\n")
    return_mean = np.mean(returns)
    np.savetxt(args.seed_result_dir + "return_mean.csv", [return_mean], fmt='%.6f')

# * ==============================  pretraining utils ============================== * #
def CrossEntropy(predict, label):
    ce = torch.nn.CrossEntropyLoss()
    label = torch.argmax(label, dim=-1)
    loss = ce(predict, label)
    return loss  # [,]

def MeanSquareError(predict, label):
    mse = torch.nn.MSELoss()
    loss = mse(predict, label)
    return loss  # [,]


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    if len(x) == 0:
        return discount_cumsum
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def get_prompt(prompt_dataset, args):
    oppo_obs_dim = oppo_obs_dim_dict[args.env_type]
    act_dim = act_dim_dict[args.env_type]
    max_steps = horizon_per_ep_dict[args.env_type]
    K = args.prompt_len
    prompt_epi = args.prompt_epi
    device = args.device
    oppo_idxs = opponent_index_dict[args.env_type]
    
    def fn(batch_size=1, oppo_pi_idx=0, max_steps=max_steps, max_len=K):
        o, a, timesteps, mask = {idx:[] for idx in oppo_idxs} , {idx:[] for idx in oppo_idxs}, {idx:[] for idx in oppo_idxs}, {idx:[] for idx in oppo_idxs}
        batch_inds = np.random.choice(
            np.arange(len(prompt_dataset[oppo_pi_idx])),
            size=(prompt_epi*batch_size),
            replace=False,
        )
        for k in range((prompt_epi*batch_size)):
            traj = prompt_dataset[oppo_pi_idx][batch_inds[k]]
            for idx in oppo_idxs:
                si = np.random.randint(0, traj[idx]['act'].shape[0])
                o[idx].append(traj[idx]['obs'][si:si + max_len].reshape(1, -1, *oppo_obs_dim))
                a[idx].append(traj[idx]['act'][si:si + max_len].reshape(1, -1, act_dim))
                timesteps[idx].append(np.arange(si, si + o[idx][-1].shape[1]).reshape(1, -1))
                timesteps[idx][-1][timesteps[idx][-1] >= max_steps] = max_steps - 1  # padding cutoff

                tlen = o[idx][-1].shape[1] # timestep length

                o[idx][-1] = np.concatenate([np.zeros((1, max_len - tlen, *oppo_obs_dim)), o[idx][-1]], axis=1)
                a[idx][-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[idx][-1]], axis=1)
                timesteps[idx][-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[idx][-1]], axis=1)
                mask[idx].append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        for idx in oppo_idxs:
            o[idx] = torch.from_numpy(np.concatenate(o[idx], axis=0)).to(dtype=torch.float32, device=device)
            a[idx] = torch.from_numpy(np.concatenate(a[idx], axis=0)).to(dtype=torch.float32, device=device)
            timesteps[idx] = torch.from_numpy(np.concatenate(timesteps[idx], axis=0)).to(dtype=torch.long, device=device)
            mask[idx] = torch.from_numpy(np.concatenate(mask[idx], axis=0)).to(device=device)
        return o, a, timesteps, mask

    return fn


def get_batch(dataset, args):
    my_obs_dim = my_obs_dim_dict[args.env_type]
    act_dim = act_dim_dict[args.env_type]
    num_oppo_policy = args.num_oppo_policy
    batch_size = args.batch_size
    K = args.history_len
    num_steps = horizon_per_ep_dict[args.env_type]
    reward_scale = reward_scale_dict[args.env_type]
    device = args.device
    my_idxs = my_index_dict[args.env_type]
    oppo_idxs = opponent_index_dict[args.env_type]
    def fn(batch_size=batch_size, max_len=K):
        o, a, a_oppo, rtg, timesteps, mask = [], [], {idx:[] for idx in oppo_idxs}, [], [], []
        for i in range(num_oppo_policy):
            batch_inds = np.random.choice(
                np.arange(len(dataset[i])),
                size=batch_size,
                replace=False,
            )
            for k in range(batch_size):
                traj = dataset[i][batch_inds[k]]
                my_traj = traj[my_idxs[0]]
                si = np.random.randint(0, my_traj['rew'].shape[0])
                o.append(my_traj['obs'][si:si + max_len].reshape(1, -1, *my_obs_dim))
                a.append(my_traj['act'][si:si + max_len].reshape(1, -1, act_dim))
                for idx in oppo_idxs:
                    a_oppo[idx].append(traj[idx]['act'][si:si + max_len].reshape(1, -1, act_dim))
                timesteps.append(np.arange(si, si + o[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= num_steps] = num_steps - 1
                rtg.append(discount_cumsum(my_traj['rew'][si:], gamma=1.)[:o[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[-1].shape[1] <= o[-1].shape[1]:
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                
                tlen = o[-1].shape[1]
                
                o[-1] = np.concatenate([np.zeros((1, max_len - tlen, *my_obs_dim)), o[-1]], axis=1)
                a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
                for idx in oppo_idxs:
                    a_oppo[idx][-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a_oppo[idx][-1]], axis=1)
                rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / reward_scale
                timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
                
        o = torch.from_numpy(np.concatenate(o, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        for idx in oppo_idxs:
            a_oppo[idx] = torch.from_numpy(np.concatenate(a_oppo[idx], axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        
        return o, a, a_oppo, rtg, timesteps, mask
    
    return fn


def flatten_prompt(prompt, batch_size, idx):
    p_o, p_a, p_timesteps, p_mask = prompt
    p_o_ = p_o[idx].reshape((batch_size, -1, *p_o[idx].shape[2:]))
    p_a_ = p_a[idx].reshape((batch_size, -1, p_a[idx].shape[-1]))
    p_timesteps_ = p_timesteps[idx].reshape((batch_size, -1))
    p_mask_ = p_mask[idx].reshape((batch_size, -1))
    return p_o_, p_a_, p_timesteps_, p_mask_


def get_prompt_batch(dataset, prompt_dataset, args):
    batch_size = args.batch_size
    num_oppo_policy = args.num_oppo_policy
    oppo_idxs = opponent_index_dict[args.env_type]

    def fn(batch_size=batch_size):
        get_prompt_fn = get_prompt(prompt_dataset, args)
        p_o, p_a, p_timesteps, p_mask = {idx:[] for idx in oppo_idxs}, {idx:[] for idx in oppo_idxs}, {idx:[] for idx in oppo_idxs}, {idx:[] for idx in oppo_idxs}
        for idx in oppo_idxs:
            p_o_list, p_a_list, p_timesteps_list, p_mask_list = [], [], [], []
            for oppo_pi_idx in range(num_oppo_policy):
                prompt_ = flatten_prompt(get_prompt_fn(batch_size, oppo_pi_idx), batch_size, idx)
                p_o_, p_a_, p_timesteps_, p_mask_ = prompt_
                p_o_list.append(p_o_)
                p_a_list.append(p_a_)
                p_timesteps_list.append(p_timesteps_)
                p_mask_list.append(p_mask_)
            p_o[idx], p_a[idx], p_timesteps[idx], p_mask[idx] = torch.cat(p_o_list, dim=0), torch.cat(p_a_list, dim=0), torch.cat(p_timesteps_list, dim=0), torch.cat(p_mask_list, dim=0)
        prompt = p_o, p_a, p_timesteps, p_mask

        get_batch_fn = get_batch(dataset, args) 
        batch = get_batch_fn(batch_size=batch_size)
        return prompt, batch

    return fn


def load_dataset(args, prompt=False):
    if prompt:
        data_path = args.load_data_path.replace("data/", "prompt/")
    else:
        data_path = args.load_data_path
    data_dict = load_pickle(data_path)
    num_oppo_policy = len(data_dict.keys())
    if not prompt:
        args.num_oppo_policy = num_oppo_policy
    my_idxs = my_index_dict[args.env_type]
    oppo_idxs = opponent_index_dict[args.env_type]
    returns_against_adv_list = [[] for _ in range(num_oppo_policy)]
    data_list = [[] for _ in range(num_oppo_policy)]
    k = 0
    for _, v in data_dict.items():
        num_epis = len(v['ep_lengths'])
        for ep in range(num_epis):
            my_o_ep = {idx:[] for idx in my_idxs}
            my_a_ep = {idx:[] for idx in my_idxs}
            my_r_ep = {idx:[] for idx in my_idxs}
            oppo_o_ep = {idx:[] for idx in oppo_idxs}
            oppo_a_ep = {idx:[] for idx in oppo_idxs}
            num_steps = v['ep_lengths'][ep]
            for step in range(num_steps):
                for idx in my_idxs:
                    my_o_ep[idx].append(np.array(v['ep_observations'][ep][step][idx]))
                    my_a_ep[idx].append(np.array(v['ep_actions'][ep][step][idx]))
                    my_r_ep[idx].append(np.array(v['ep_rewards'][ep][step]))
                for idx in oppo_idxs:
                    oppo_o_ep[idx].append(np.array(v['ep_observations'][ep][step][idx]))
                    oppo_a_ep[idx].append(np.array(v['ep_actions'][ep][step][idx]))
            data_dict_ep = dict()
            for idx in my_idxs:
                data_dict_ep[idx] = {
                    "obs": np.array(my_o_ep[idx]),
                    "act": np.array(my_a_ep[idx]),
                    "rew": np.array(my_r_ep[idx]),
                }
            for idx in oppo_idxs:
                data_dict_ep[idx] = {
                    "obs": np.array(oppo_o_ep[idx]),
                    "act": np.array(oppo_a_ep[idx]),
                }
            data_list[k].append(data_dict_ep)
            returns_against_adv_list[k].append(np.sum(v['ep_rewards'][ep]))
        k += 1
    return data_list


# * ============================== model loading utils ============================== * #

def get_trailing_number(s):
    """
    Get the trailing number from a string,
    i.e. 'file123' -> '123'
    """
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def get_max_iter(agent_folder):
    """Return biggest PBT iteration that has been run"""
    saved_iters = []
    for folder_s in os.listdir(agent_folder):
        folder_iter = get_trailing_number(folder_s) 
        if folder_iter is not None:
            saved_iters.append(folder_iter)
    if len(saved_iters) == 0:
        raise ValueError("Agent folder {} seemed to not have any pbt_iter subfolders".format(agent_folder))
    return max(saved_iters)

def get_model_policy(step_fn, sim_threads, is_joint_action=False):
    """
    Returns the policy function `p(s, index)` from a saved model at `save_dir`.
    
    step_fn: a function that takes in observations and returns the corresponding
             action probabilities of the agent
    """
    def encoded_state_policy(observations, stochastic=True, return_action_probs=False):
        """Takes in SIM_THREADS many losslessly encoded states and returns corresponding actions"""
        action_probs_n = step_fn(observations)
        action_dim = action_probs_n.shape[-1]

        if return_action_probs:
            return action_probs_n
        
        if stochastic:
            action_idxs = [np.random.choice(action_dim, p=action_probs) for action_probs in action_probs_n]
        else:
            action_idxs = [np.argmax(action_probs) for action_probs in action_probs_n]

        return np.array(action_idxs)

    def state_policy(mdp_state, mdp, agent_index, stochastic=True, return_action_probs=False):
        """Takes in a Overcooked state object and returns the corresponding action"""
        obs = mdp.lossless_state_encoding(mdp_state)[agent_index]
        padded_obs = np.array([obs] + [np.zeros(obs.shape)] * (sim_threads - 1))
        action_probs = step_fn(padded_obs)[0] # Discards all padding predictions

        if return_action_probs:
            return action_probs

        if stochastic:
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else:
            action_idx = np.argmax(action_probs)

        if is_joint_action:
            # NOTE: Probably will break for this case, untested
            action_idxs = Action.INDEX_TO_ACTION_INDEX_PAIRS[action_idx]
            joint_action = [Action.INDEX_TO_ACTION[i] for i in action_idxs]
            return joint_action

        return Action.INDEX_TO_ACTION[action_idx]

    return state_policy, encoded_state_policy

def get_model_policy_from_saved_model(save_dir, sim_threads):
    """Get a policy function from a saved model"""
    import tensorflow as tf
    predictor = tf.contrib.predictor.from_saved_model(save_dir)
    step_fn = lambda obs: predictor({"obs": obs})["action_probs"]
    return get_model_policy(step_fn, sim_threads)

def get_agent_from_saved_model(save_dir, sim_threads):
    """Get Agent corresponding to a saved model"""
    # NOTE: Could remove dependency on sim_threads if get the sim_threads from config or dummy env
    state_policy, processed_obs_policy = get_model_policy_from_saved_model(save_dir, sim_threads)
    return AgentFromPolicy(state_policy, processed_obs_policy)

def get_onnx_agent_model(agent_to_load_path, sim_threads, device, get_prob):
    onnx_model_path = os.path.join(agent_to_load_path, "model.onnx")
    if "cuda" in device:
        providers = [('CUDAExecutionProvider', {"device_id": device.split(":")[-1]})]
    elif "cpu" in device:
        providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    def step_fn(obs):
        padded_obs = np.array([obs] + [np.zeros(obs.shape)] * (sim_threads - 1)).astype(np.float32)
        ort_inputs = {
            "obs": padded_obs
        }
        ort_outs = ort_session.run(["action_probs"], ort_inputs)
        action_probs = ort_outs[0][0]
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        if get_prob:
            act_prob = action_probs[action_idx]
            return action_idx, act_prob
        return action_idx
    return step_fn

def get_trt_agent_model(agent_to_load_path, sim_threads, device, get_prob, env_type):
    import onnx_tensorrt.backend as backend
    model_path = os.path.join(agent_to_load_path, "model.onnx")
    model = onnx.load(model_path)
    engine = backend.prepare(model, device=device)
    
    oppo_obs_dim = oppo_obs_dim_dict[env_type]
    obs_ = np.random.randn(sim_threads, *oppo_obs_dim).astype(np.float32, order='C')
    input_data_ = tuple([obs_])
    engine.run(input_data_)
    
    def step_fn(obs):
        padded_obs = np.array([obs] + [np.zeros(oppo_obs_dim)] * (sim_threads - 1)).astype(np.float32, order='C')
        input_data = {
            'obs': padded_obs,
        }
        tensorrt_outputs = engine.engine.run(input_data)
        action_probs = tensorrt_outputs[1][0]
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        if get_prob:
            act_prob = action_probs[action_idx]
            return action_idx, act_prob
        return action_idx
    return step_fn

def get_pbt_agent_from_config(save_dir=None, sim_threads=0, seed=0, agent_idx=0, best=False, agent_to_load_path=None, model_type="tf", device="cpu", get_prob=False, env_type="oc"):
    if agent_to_load_path is None:
        agent_folder = save_dir + 'seed_{}/agent{}'.format(seed, agent_idx)
        if best:
            agent_to_load_path = agent_folder  + "/best"
        else:
            agent_to_load_path = agent_folder  + "/pbt_iter" + str(get_max_iter(agent_folder))
    if model_type == "trt":
        agent = get_trt_agent_model(agent_to_load_path, sim_threads, device.upper(), get_prob, env_type)
    elif model_type == "onnx":
        agent = get_onnx_agent_model(agent_to_load_path, sim_threads, device, get_prob)
    elif model_type == "tf":
        agent = get_agent_from_saved_model(agent_to_load_path, sim_threads)
    return agent

# * ============================== evaluation utils ============================== * #

class AgentGroupObs(object):
    """
    AgentGroup is a group of N agents used to sample 
    joint actions in the context of an OvercookedEnv instance.
    """

    def __init__(self, *agents, sim_threads):
        self.agents = agents
        self.sim_threads = sim_threads
        self.n = len(self.agents)

    def joint_action(self, obs):
        # 调用每个智能体(pi)，用对应的观测(obs[i])，返回联合动作
        return tuple(pi(obs[i]) for i, pi in enumerate(self.agents))

def run_agents(env, env_type, agent_group, agent_idx, include_final_state):
    EP_LEN = horizon_per_ep_dict[env_type]
    ACT_DIM = act_dim_dict[env_type]
    trajectory = []
    done = False
    t = 0
    if env_type == "oc":
        cumulative_sparse_rewards = [0., 0.]
    elif env_type == "lbf":
        cumulative_sparse_rewards = [0., 0.]
    elif env_type == "pp":
        cumulative_sparse_rewards = [0., 0., 0., 0.]
    if env_type == "oc":
        env.reset()
        o_t = env.mdp.lossless_state_encoding(env.state)
    else:
        o_t = env.reset()
    while not done:
        a_t_idx = agent_group.joint_action(o_t)

        # Break if either agent is out of actions
        if any([a is None for a in a_t_idx]):
            break
        
        if env_type == "oc":
            a_t_ = []
            for a in a_t_idx:
                a_t_.append(Action.INDEX_TO_ACTION[a])
            a_t = tuple(a_t_)
        elif env_type == "lbf":
            a_t = a_t_idx
        elif env_type == "pp":
            a_t_ = []
            for a in a_t_idx:
                a_t_.append(np.eye(ACT_DIM)[a])
            a_t = tuple(a_t_)

        s_tp1, r_t_, _, _ = env.step(a_t)
        if env_type == "oc":
            r_t = (r_t_, r_t_)
        else:
            r_t = r_t_
        t += 1
        for i in range(len(cumulative_sparse_rewards)):
            cumulative_sparse_rewards[i] += r_t[i]
        
        if t >= EP_LEN:
            done = True
        else:
            done = False
        
        a_t_onehot_ = []
        for a in a_t_idx:
            a_t_onehot_.append(np.eye(ACT_DIM)[a])
        a_t_onehot = tuple(a_t_onehot_)
        trajectory.append((o_t, a_t_onehot, r_t[agent_idx], done))
        if env_type == "oc":
            o_t = env.mdp.lossless_state_encoding(s_tp1)
        else:
            o_t = s_tp1

    assert len(trajectory) == t, "{} vs {}".format(len(trajectory), t)
    
    # Add final state
    if include_final_state:
        if env_type == "oc":
            a_t_ = (None, None)
            o_tp1 = env.mdp.lossless_state_encoding(s_tp1)
        elif env_type == "lbf":
            a_t_ = (None, None)
            o_tp1 = s_tp1
        elif env_type == "pp":
            a_t_ = (None, None, None, None)
            o_tp1 = s_tp1
        trajectory.append((o_tp1, a_t_, 0.0, True))
    return np.array(trajectory), t, np.array(cumulative_sparse_rewards)


def get_rollouts(env, env_type, agent_group, num_games, agent_idx, final_state=False):
    
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
        trajectory, time_taken, tot_rews_sparse = run_agents(env, env_type, agent_group, agent_idx, include_final_state=final_state)
        obs, actions, rews, dones = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3]
        trajectories["ep_observations"].append(obs)
        trajectories["ep_actions"].append(actions)
        trajectories["ep_rewards"].append(rews)
        trajectories["ep_dones"].append(dones)
        trajectories["ep_returns"].append(tot_rews_sparse)
        trajectories["ep_returns_sparse"].append(tot_rews_sparse)
        trajectories["ep_lengths"].append(time_taken)
        env.reset()

    trajectories = {k: np.array(v) for k, v in trajectories.items()}
    return trajectories