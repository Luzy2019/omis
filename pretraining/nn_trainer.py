import numpy as np
# import onnx
import torch
from torch import nn
# import onnxruntime as ort
import time
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pretraining.utils import LOG, opponent_index_dict, my_obs_dim_dict, oppo_obs_dim_dict, act_dim_dict

class Trainer:
    def __init__(self,
                model,
                optimizer,
                scheduler,
                get_prompt_batch_fn,
                get_prompt_fn,
                act_loss_fn,
                value_loss_fn,
                args):
        self.model = model
        self.batch_size = args.batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.get_prompt_batch_fn = get_prompt_batch_fn
        self.get_prompt_fn = get_prompt_fn
        self.act_loss_fn = act_loss_fn
        self.value_loss_fn = value_loss_fn
        
        self.clip_grad = args.clip_grad
        self.vf_coef = args.vf_coef
        self.oppo_pi_coef = args.oppo_pi_coef
        self.args = args
        self.diagnostics = dict()
        
        self.oppo_idxs = opponent_index_dict[args.env_type]
        self.argsorted_oppo_idxs = np.argsort(self.oppo_idxs)

        self.start_time = time.time()

    def train(self, num_update):
        act_losses, value_losses, oppo_pi_losses = [], [], []
        logs = dict()

        train_start = time.time()
        self.model.train()
        for _ in range(num_update):
            act_loss, value_loss, oppo_pi_loss = self.train_step()
            act_losses.append(act_loss)
            value_losses.append(value_loss)
            oppo_pi_losses.append(oppo_pi_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        action_loss_mean = np.mean(act_losses)
        value_loss_mean = np.mean(value_losses)
        oppo_pi_loss_mean = np.mean(oppo_pi_losses)
        
        logs['time/training'] = time.time() - train_start
        logs['training/action_loss_mean'] = action_loss_mean
        logs['training/action_loss_std'] = np.std(act_losses)
        logs['training/value_loss_mean'] = value_loss_mean
        logs['training/value_loss_std'] = np.std(value_losses)
        logs['training/oppo_pi_loss_mean'] = oppo_pi_loss_mean
        logs['training/oppo_pi_loss_std'] = np.std(oppo_pi_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs, action_loss_mean, value_loss_mean, oppo_pi_loss_mean

    def train_step(self):
        prompt, batch = self.get_prompt_batch_fn()
        o, a, a_oppo, rtg, timesteps, mask = batch
        a_target = a.detach().clone()
        v_target = rtg.detach().clone()

        a_oppo_target = []
        for i in self.argsorted_oppo_idxs:
            a_oppo_target.append(a_oppo[self.oppo_idxs[i]].detach().clone())
        
        oppo_actions, prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask = [], [], [], [], []
        for i in self.argsorted_oppo_idxs:
            oppo_actions.append(a_oppo[self.oppo_idxs[i]])
            prompt_states.append(prompt[0][self.oppo_idxs[i]])
            prompt_actions.append(prompt[1][self.oppo_idxs[i]])
            prompt_timesteps.append(prompt[2][self.oppo_idxs[i]])
            prompt_attention_mask.append(prompt[3][self.oppo_idxs[i]])
        a_oppo = torch.stack(oppo_actions, dim=0).to(dtype=torch.float32, device=o.device)
        p_o = torch.stack(prompt_states, dim=0).to(dtype=torch.float32, device=o.device)
        p_a = torch.stack(prompt_actions, dim=0).to(dtype=torch.float32, device=o.device)
        p_t = torch.stack(prompt_timesteps, dim=0).to(dtype=torch.long, device=o.device)
        p_m = torch.stack(prompt_attention_mask, dim=0).to(device=o.device)
        
        a_preds, v_preds, a_oppo_preds = self.model.forward(
            o, a, a_oppo, rtg[:, :-1], timesteps, mask,
            p_o, p_a, p_t, p_m,
        )

        act_dim = a_preds.shape[2]
        a_preds = a_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        a_target = a_target.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        
        v_preds = v_preds.reshape(-1, 1)[mask.reshape(-1) > 0]
        v_target = v_target[:, :-1].reshape(-1, 1)[mask.reshape(-1) > 0]
        
        a_oppo_preds_list = []
        for i in range(len(self.oppo_idxs)):
            act_dim = a_oppo_preds[i].shape[2]
            a_oppo_preds_list.append(a_oppo_preds[i].reshape(-1, act_dim)[mask.reshape(-1) > 0])
            a_oppo_target[i] = a_oppo_target[i].reshape(-1, act_dim)[mask.reshape(-1) > 0]

        act_loss = self.act_loss_fn(a_preds, a_target)
        value_loss = self.value_loss_fn(v_preds, v_target)
        oppo_pi_loss = 0
        for i in range(len(self.oppo_idxs)):
            oppo_pi_loss += self.act_loss_fn(a_oppo_preds_list[i], a_oppo_target[i])
        loss = act_loss + self.vf_coef * value_loss + self.oppo_pi_coef * oppo_pi_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/total_loss'] = loss.detach().clone().cpu().item()

        return act_loss.detach().cpu().item(), value_loss.detach().cpu().item(), oppo_pi_loss.detach().cpu().item()
    
    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        LOG.info(f'Model saved to: {model_path}')
    
    # def save_onnx_model(self, model_path):
    #     self.model.eval()
        
    #     BS = 1
    #     obs_dim = my_obs_dim_dict[self.args.env_type]
    #     act_dim = act_dim_dict[self.args.env_type]
    #     oppo_obs_dim = oppo_obs_dim_dict[self.args.env_type]
    #     oppo_idxs = self.oppo_idxs
    #     K = self.args.history_len
    #     PK = self.args.prompt_len*self.args.prompt_epi
        
    #     obs = torch.randn(BS, K, *obs_dim, requires_grad=True)
    #     actions = torch.randn(BS, K, act_dim, requires_grad=True)
    #     oppo_actions = []
    #     for _ in range(len(oppo_idxs)):
    #         oppo_actions.append(torch.randn(BS, K, act_dim, requires_grad=True))
    #     oppo_actions = torch.stack(oppo_actions, dim=0)
    #     returns_to_go = torch.randn(BS, K, 1, requires_grad=True)
    #     timesteps = torch.ones(BS, K, dtype=torch.long)
    #     attention_mask = torch.ones(BS, K, dtype=torch.long)
    #     prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask = [], [], [], []
    #     for _ in range(len(oppo_idxs)):
    #         prompt_states.append(torch.randn(BS, PK, *oppo_obs_dim, requires_grad=True))
    #         prompt_actions.append(torch.randn(BS, PK, act_dim, requires_grad=True))
    #         prompt_timesteps.append(torch.ones(BS, PK, dtype=torch.long))
    #         prompt_attention_mask.append(torch.ones(BS, PK, dtype=torch.long))
    #     prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask = \
    #         torch.stack(prompt_states, dim=0), torch.stack(prompt_actions, dim=0), torch.stack(prompt_timesteps, dim=0), torch.stack(prompt_attention_mask, dim=0)
    #     torch_input = obs.to(device=self.args.device, dtype=torch.float32), actions.to(device=self.args.device, dtype=torch.float32), oppo_actions.to(device=self.args.device, dtype=torch.float32), returns_to_go.to(device=self.args.device, dtype=torch.float32), timesteps.to(device=self.args.device, dtype=torch.long), attention_mask.to(device=self.args.device), prompt_states.to(device=self.args.device, dtype=torch.float32), prompt_actions.to(device=self.args.device, dtype=torch.float32), prompt_timesteps.to(device=self.args.device, dtype=torch.long), prompt_attention_mask.to(device=self.args.device)
        
    #     onnx_model_path = f"{model_path}.onnx"
    #     torch.onnx.export(
    #         model=self.model,
    #         args=torch_input,
    #         f=onnx_model_path,
    #         export_params=True,
    #         input_names = ['obs', 'actions', 'oppo_actions', 'returns_to_go', 'timesteps', 'attention_mask', 'prompt_states', 'prompt_actions', 'prompt_timesteps', 'prompt_attention_mask'],
    #         output_names = ['action_preds', 'value_preds', 'oppo_action_preds'],
    #         dynamic_axes={
    #             'obs': {0: 'batch_size', 1:'history_len'},
    #             'actions': {0: 'batch_size', 1:'history_len'},
    #             'oppo_actions': {0: 'num_oppo', 1:'batch_size', 2:'history_len'},
    #             'returns_to_go': {0: 'batch_size', 1:'history_len'},
    #             'timesteps': {0: 'batch_size', 1:'history_len'},
    #             'attention_mask': {0: 'batch_size', 1:'history_len'},
    #             'prompt_states': {0: 'num_oppo', 1:'batch_size', 2:'prompt_len'},
    #             'prompt_actions': {0: 'num_oppo', 1:'batch_size', 2:'prompt_len'},
    #             'prompt_timesteps': {0: 'num_oppo', 1:'batch_size', 2:'prompt_len'},
    #             'prompt_attention_mask': {0: 'num_oppo', 1:'batch_size', 2:'prompt_len'},
    #             'action_preds': {0: 'batch_size', 1:'history_len'},
    #             'value_preds': {0: 'batch_size', 1:'history_len'},
    #             'oppo_action_preds': {0: 'num_oppo', 1:'batch_size', 2:'history_len'},
    #         }
    #     )