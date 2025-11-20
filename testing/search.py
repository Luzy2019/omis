import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import copy
from typing import List, Tuple, Union, Callable, Dict, Any

import numpy as np
from scipy.special import softmax
from pretraining.utils import my_index_dict, opponent_index_dict, act_dim_dict, my_obs_dim_dict
from overcooked_ai_py.mdp.actions import Action

reward_scale_dict = {
    "oc": 10.,
    "lbf": 0.1,
    "pp": 10.,
}

class Node(object):
    def __init__(self, cur_model_info = None, parent_oppo_actions = None, parent_value = None, parent: "Node" = None, cur_real_deepth = None, prior_p: float = 1.0) -> None:
        self._cur_model_info = cur_model_info
        self._parent_oppo_actions = parent_oppo_actions
        self._parent_value = parent_value
        self._parent = parent
        self._cur_real_deepth = cur_real_deepth
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p

    @property
    def value(self) -> float:
        if self._visit_count == 0:
            return 0
        return self._value_sum / self._visit_count

    def update(self, cum_reward) -> None:
        self._visit_count += 1
        self._value_sum += cum_reward

    def update_recursive(self, cum_rewards) -> None:
            cum_reward = cum_rewards.pop(-1)
            self.update(cum_reward)
            if self.is_root():
                return
            self._parent.update_recursive(cum_rewards)

    def is_leaf(self) -> bool:
        return self._children == {}

    def is_root(self) -> bool:
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count
    
    @property
    def cur_model_info(self) -> None:
        return self._cur_model_info
    
    @cur_model_info.setter
    def cur_model_info(self, cur_model_info) -> None:
        self._cur_model_info = cur_model_info
    
    @property
    def parent_oppo_actions(self) -> None:
        return self._parent_oppo_actions
    
    @property
    def parent_value(self) -> None:
        return self._parent_value
    
    @property
    def cur_real_deepth(self) -> None:
        return self._cur_real_deepth


class Searcher(object):
    def __init__(self, args_dict, fake_env, pi_fn: Callable) -> None:
        self._args = args_dict
        
        self.env_type = self._args['env_type']
        self.history_len = self._args['history_len']
        
        self.num_rollout_per_action = self._args['num_rollout_per_action']
        self.rollout_length = self._args['rollout_length']
        self.search_gamma = self._args['search_gamma']
        
        self.fake_env = fake_env
        self.pi_fn = pi_fn
        self.my_idxs = my_index_dict[self.env_type]
        self.oppo_idxs = opponent_index_dict[self.env_type]
        self.argsorted_oppo_idxs = np.argsort(self.oppo_idxs)
        self.my_obs_dim = my_obs_dim_dict[self.env_type]
        self.act_dim = act_dim_dict[self.env_type]
        self.reward_scale = reward_scale_dict[self.env_type]
    
    @property
    def legal_actions(self):
        return range(self.act_dim)
    
    def search(
            self,
            statedict: Dict[str, Any],
            cur_model_info,
            sample: bool = True
    ) -> Tuple[int, List[float]]:

        # 当前真实环境的步数
        real_env_cur_t = statedict['t']
        # 创建根节点，传入当前模型信息和当前步数
        root = Node(cur_model_info=cur_model_info, cur_real_deepth=real_env_cur_t)

        # 根据环境类型同步/复制fake_env
        if self.env_type == 'oc':
            self.fake_env.load_from_statedict(statedict)
        elif self.env_type in ['lbf', 'pp']:
            self.fake_env = copy.deepcopy(statedict['root_env'])
        # 扩展根节点，获得动作概率和原始价值预测
        action_probs, original_value_preds = self._expand_leaf_node(root)

        hasrewards = []  # 用于统计各次模拟是否得到非零奖励
        nonzero_cum_rewards = dict()  # 记录每个动作下获得非零奖励的累计奖励
        for action in range(self.act_dim):
            nonzero_cum_rewards_ = []
            for _ in range(self.num_rollout_per_action):
                # 每次rollout都要同步fake_env到指定状态
                if self.env_type == 'oc':
                    self.fake_env.load_from_statedict(statedict)
                elif self.env_type in ['lbf', 'pp']:
                    self.fake_env = copy.deepcopy(statedict['root_env'])
                # 对指定动作做一次rollout模拟，返回当前动作是否有奖励以及累计奖励
                hasreward, cum_rewards = self._simulate(action, root, self.fake_env)
                if hasreward:
                    nonzero_cum_rewards_.append(cum_rewards)
                hasrewards.append(hasreward)
            nonzero_cum_rewards[action] = nonzero_cum_rewards_
        hasreward = any(hasrewards)  # 统计所有模拟过程中是否有任何非零奖励出现
        
        # 统计每个动作的子节点value（如果没有则为0）
        action_values = []
        for action in range(self.act_dim):
            if action in root.children:
                action_values.append((action, root.children[action].value[0]))
            else:
                action_values.append((action, 0))

        actions, values = zip(*action_values)

        values_np = np.array(values, dtype=np.float32, order='C')
        values_np_sum = values_np.sum()
        # 如果所有动作value都为0，则均匀分布。否则，归一化value为概率分布
        if values_np_sum == 0:
            value_probs = np.ones_like(values_np, dtype=np.float32, order='C') / self.act_dim
        else:
            if self.env_type == 'pp':
                value_probs = values_np
            else:
                value_probs = values_np / values_np_sum
        
        # 如果所有rollout都没拿到奖励，则采用模型动作概率；否则用rollout value
        if hasreward:
            final_action_probs = value_probs
            value_preds = original_value_preds
        else:
            final_action_probs = action_probs
            value_preds = original_value_preds

        # 根据hasreward和环境类型决定采用的动作样本方式
        if hasreward:
            if self.env_type == 'pp':
                # PP环境下对最大概率动作均匀采样
                max_actions = np.argwhere(final_action_probs==np.max(final_action_probs)).flatten()
                action = np.random.choice(max_actions)
            else:
                action = actions[np.argmax(final_action_probs)]
        else:
            if sample:
                action = np.random.choice(actions, p=final_action_probs)
            else:
                action = actions[np.argmax(final_action_probs)]
        
        # 返回所选动作和动作价值预测
        return action, value_preds

    def _process_model_info(self, old_model_info, my_obs, action, o_actions, new_t, last_value):
        # 拆分出旧模型的信息
        obs, actions, oppo_actions, returns_to_go, timesteps, attention_mask, o_oppo, a_oppo, timestep_oppo, mask_oppo = old_model_info
        
        # 处理我方观测，添加到历史轨迹（保持最大长度为history_len）
        my_obs = my_obs.reshape(1, 1, *self.my_obs_dim).astype(dtype=np.float32, order='C')
        new_obs = np.concatenate((obs, my_obs), axis=1)
        new_obs = new_obs[:, -self.history_len:]

        # 处理我方动作，将当前动作one-hot化，并添加到动作轨迹
        my_action = np.eye(self.act_dim, dtype=np.float32, order='C')[action]
        actions[0, -1] = my_action  # 替换最后一步（可能是padding）
        padding_action = np.zeros((1, 1, self.act_dim), dtype=np.float32, order='C')  # 新的padding占位
        new_actions = np.concatenate((actions, padding_action), axis=1)
        new_actions = new_actions[:, -self.history_len:]
        
        # 处理对方动作，循环处理所有对手，下标一一对应
        for i in range(len(self.oppo_idxs)):
            o_action = np.eye(self.act_dim, dtype=np.float32, order='C')[o_actions[i]]
            oppo_actions[i, 0, -1] = o_action
        padding_o_action = np.zeros((len(self.oppo_idxs), 1, 1, self.act_dim), dtype=np.float32, order='C')
        new_oppo_actions = np.concatenate((oppo_actions, padding_o_action), axis=2)
        new_oppo_actions = new_oppo_actions[:, :, -self.history_len:]

        # 处理returns_to_go，更新最后一步为last_value（bootstrap/critic估计的下一状态价值），添加padding
        returns_to_go[0, -1] = last_value
        padding_return = np.zeros((1, 1, 1), dtype=np.float32, order='C')
        new_returns_to_go = np.concatenate((returns_to_go, padding_return), axis=1)
        new_returns_to_go = new_returns_to_go[:, -self.history_len:]

        # 处理timestep，添加当前timestep
        padding_timestep = np.ones((1, 1), dtype=np.int64, order='C') * new_t
        new_timesteps = np.concatenate((timesteps, padding_timestep), axis=1)
        new_timesteps = new_timesteps[:, -self.history_len:]

        # 处理mask，补全新的有效长度
        padding_mask = np.ones((1, 1), dtype=np.int64, order='C')
        new_attention_mask = np.concatenate((attention_mask, padding_mask), axis=1)
        new_attention_mask = new_attention_mask[:, -self.history_len:]

        # 打包所有处理完的序列，作为新的model info
        cur_model_info = (
            new_obs,               # 新观测序列
            new_actions,           # 新动作序列
            new_oppo_actions,      # 新对手动作序列
            new_returns_to_go,     # 新rtg序列
            new_timesteps,         # 新timesteps
            new_attention_mask,    # 新mask
            o_oppo, a_oppo, timestep_oppo, mask_oppo  # 其它信息保持不变
        )
        
        return cur_model_info
    
    def _cumsum_reward(self, rewards, leaf_value, gamma):
        cum_rewards = []
        cum_reward = leaf_value
        for reward in rewards[::-1]:
            cum_reward = reward + gamma * cum_reward
            cum_rewards.append(cum_reward)
        cum_rewards.append(cum_reward)
        cum_rewards = cum_rewards[::-1]
        return cum_rewards
    
    def _fake_step(self, node, fake_env, action):
        act_n = [None for _ in self.my_idxs+self.oppo_idxs]
        o_actions = node.parent_oppo_actions
        if self.env_type == 'oc':
            my_act = Action.INDEX_TO_ACTION[action]
            o_act = Action.INDEX_TO_ACTION[o_actions[0]]
            act_n[self.my_idxs[0]] = my_act
            act_n[self.oppo_idxs[0]] = o_act
            next_state, reward, done, _ = fake_env.step(act_n)
        elif self.env_type == 'lbf':
            my_act = action
            o_act = o_actions[0]
            act_n[self.my_idxs[0]] = my_act
            act_n[self.oppo_idxs[0]] = o_act
            next_obs_n, reward_n, done_n, _ = fake_env.step(act_n)
            reward = reward_n[self.my_idxs[0]]
            done = any(done_n)
        elif self.env_type == 'pp':
            my_act = np.eye(self.act_dim, dtype=np.float32, order='C')[action]
            act_n[self.my_idxs[0]] = my_act
            for i in self.argsorted_oppo_idxs:
                o_act = np.eye(self.act_dim, dtype=np.float32, order='C')[o_actions[i]]
                act_n[self.oppo_idxs[i]] = o_act
            next_obs_n, reward_n, _, _ = fake_env.step(act_n)
            reward = reward_n[self.my_idxs[0]]
            new_t = node.cur_real_deepth
            done = new_t >= 100
        if node.cur_model_info is None:
            if self.env_type == 'oc':
                next_obs_n = fake_env.mdp.lossless_state_encoding(next_state)
                new_t = fake_env.t
            elif self.env_type == 'lbf':
                new_t = fake_env.current_step
            my_obs = next_obs_n[self.my_idxs[0]]
            last_value = node.parent_value
            old_model_info = node.parent.cur_model_info
            cur_model_info = self._process_model_info(old_model_info, my_obs, action, o_actions, new_t, last_value)
            node.cur_model_info = cur_model_info
        return reward, done
    
    def _simulate(self, first_action, node, fake_env) -> bool:
        hasreward = False
        rewards = []
        node = node.children[first_action]
        reward, done = self._fake_step(node, fake_env, first_action)
        rewards.append(reward / self.reward_scale)
        if not done:
            for _ in range(self.rollout_length):
                if node.is_leaf():
                    self._expand_leaf_node(node)
                action, node = self._select_child(node)
                if action is None:
                    break
                reward, done = self._fake_step(node, fake_env, action)
                rewards.append(reward / self.reward_scale)
                if done:
                    break

        if node.is_leaf():
            _, leaf_value = self._expand_leaf_node(node)
        else:
            rand_act = np.random.choice(self.act_dim)
            leaf_value = node.children[rand_act].parent_value

        cum_rewards = self._cumsum_reward(rewards, leaf_value, gamma=self.search_gamma)
        
        cum_rewards_ = np.concatenate(cum_rewards, axis=0).tolist()
        node.update_recursive(cum_rewards)
        if self.env_type in ['oc', 'lbf']:
            hasreward = True if sum(rewards) > (0 / self.reward_scale) else False
        elif self.env_type == 'pp':
            hasreward = True if sum(rewards) <= (-10 / self.reward_scale) else False
        return hasreward, cum_rewards_

    def _select_child(self, node) -> Tuple[Union[int, float], Node]:
        action = None
        child = None
        action_probs = []
        for action_tmp, child_tmp in node.children.items():
            if action_tmp in self.legal_actions:
                action_probs.append((action_tmp, child_tmp.prior_p))
        actions, probs = zip(*action_probs)
        probs = np.array(probs, dtype=np.float32, order='C')
        action = np.random.choice(actions, p=probs)
        child = node.children[action]
        if child is None:
            child = node

        return action, child

    def _expand_leaf_node(self, node) -> float:
        action_preds, value_preds, oppo_action_preds = self.pi_fn(*node.cur_model_info)
        if self.env_type in ['oc', 'lbf']:
            if value_preds[0] < 0:
                value_preds[0] = 0.0
        action_preds = softmax(action_preds)
        
        parent_oppo_actions = []
        for o_i in range(len(oppo_action_preds)):
            o_act = softmax(oppo_action_preds[o_i])
            o_act_idx = np.random.choice(self.act_dim, p=o_act)
            parent_oppo_actions.append(o_act_idx)
        
        for action, prior_p in enumerate(action_preds):
            if action in self.legal_actions:
                node.children[action] = Node(parent_oppo_actions=parent_oppo_actions, parent_value=value_preds, parent=node, cur_real_deepth=node.cur_real_deepth+1, prior_p=prior_p)

        return action_preds, value_preds