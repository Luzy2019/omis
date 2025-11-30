import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from pretraining.trajectory_gpt2 import GPT2Model
from pretraining.utils import my_obs_dim_dict, oppo_obs_dim_dict, act_dim_dict, horizon_per_ep_dict, opponent_index_dict, my_index_dict


class GPTModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.obs_dim = my_obs_dim_dict[args.env_type]
        self.act_dim = act_dim_dict[args.env_type]
        self.history_len = args.history_len
        self.hidden_size = args.hidden_dim
        args_dict = args.__dict__
        config = transformers.GPT2Config(vocab_size=1, n_embd=self.hidden_size, **args_dict)

        # NOTE: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.gpt_decoder = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.gpt_decoder.parallelize()
        self.oppo_obs_dim = oppo_obs_dim_dict[args.env_type]
        self.oppo_idxs = opponent_index_dict[args.env_type]
        self.argsorted_oppo_idxs = np.argsort(self.oppo_idxs).tolist()
        self.my_idxs = my_index_dict[args.env_type]

        if self.args.env_type == "Harfang":
            max_ep_len = (horizon_per_ep_dict[args.env_type])
        else:
            max_ep_len = (horizon_per_ep_dict[args.env_type] + 20)

        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        if self.args.env_type == "oc":
            self.embed_state = torch.nn.Sequential(
                nn.Conv2d(self.obs_dim[-1], 25, 5, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "valid"),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(25 * 3 * 2, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif self.args.env_type in ["lbf", "pp", "Harfang"]:
            self.embed_state = nn.Sequential(
                nn.Linear(*self.obs_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        if self.args.env_type in ["oc", "lbf", "Harfang"]:
            self.embed_oppo_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        elif self.args.env_type == "pp":
            self.embed_oppo_action0 = torch.nn.Linear(self.act_dim, self.hidden_size)
            self.embed_oppo_action1 = torch.nn.Linear(self.act_dim, self.hidden_size)
            self.embed_oppo_action2 = torch.nn.Linear(self.act_dim, self.hidden_size)
        
        if self.args.env_type == "Harfang":
            self.embed_agent_idx = nn.Embedding(len(self.oppo_idxs)+len(self.my_idxs)+2, self.hidden_size)
        else:
            self.embed_agent_idx = nn.Embedding(len(self.oppo_idxs)+len(self.my_idxs)+2, self.hidden_size)
        
        self.prompt_embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)

        if self.args.env_type == "oc":
            self.prompt_embed_state = torch.nn.Sequential(
                nn.Conv2d(self.oppo_obs_dim[-1], 25, 5, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "same"),
                nn.LeakyReLU(),
                nn.Conv2d(25, 25, 3, 1, "valid"),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(25 * 3 * 2, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif self.args.env_type == "lbf":
            self.prompt_embed_state = nn.Sequential(
                nn.Linear(*self.oppo_obs_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif self.args.env_type == "pp":
            self.prompt_embed_state0 = nn.Sequential(
                nn.Linear(*self.oppo_obs_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.prompt_embed_state1 = nn.Sequential(
                nn.Linear(*self.oppo_obs_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.prompt_embed_state2 = nn.Sequential(
                nn.Linear(*self.oppo_obs_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        elif self.args.env_type == "Harfang":
            self.prompt_embed_state = nn.Sequential(
                nn.Linear(*self.oppo_obs_dim, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        if self.args.env_type in ["oc", "lbf", "Harfang"]:
            self.prompt_embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        elif self.args.env_type == "pp":
            self.prompt_embed_action0 = torch.nn.Linear(self.act_dim, self.hidden_size)
            self.prompt_embed_action1 = torch.nn.Linear(self.act_dim, self.hidden_size)
            self.prompt_embed_action2 = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if args.action_tanh else []))
        )
        self.predict_value = torch.nn.Linear(self.hidden_size, 1)
        if self.args.env_type in ["oc", "lbf", "Harfang"]:
            self.predict_oppo_action = nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if args.action_tanh else []))
            )
        elif self.args.env_type == "pp":
            self.predict_oppo_action0 = nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if args.action_tanh else []))
            )
            self.predict_oppo_action1 = nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if args.action_tanh else []))
            )
            self.predict_oppo_action2 = nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if args.action_tanh else []))
            )

    def forward(self, obs, actions, oppo_actions, returns_to_go, timesteps, attention_mask, prompt_states, prompt_actions, prompt_timesteps, prompt_attention_mask):
        batch_size, seq_length = obs.shape[0], obs.shape[1]

        # embed each modality with a different head
        if self.args.env_type == "oc":
            obs = obs.permute(0, 1, 4, 2, 3).reshape(-1, self.obs_dim[-1], self.obs_dim[0], self.obs_dim[1])
            state_embeddings = self.embed_state(obs).reshape(batch_size, seq_length, self.hidden_size)
        elif self.args.env_type in ["lbf", "pp", "Harfang"]:
            state_embeddings = self.embed_state(obs)
        action_embeddings = self.embed_action(actions)
        
        if self.args.env_type in ["oc", "lbf", "Harfang"]:
            oppo_action_embeddings = self.embed_oppo_action(oppo_actions[0])
        elif self.args.env_type == "pp":
            oppo_action_embeddings0 = self.embed_oppo_action0(oppo_actions[0])
            oppo_action_embeddings1 = self.embed_oppo_action1(oppo_actions[1])
            oppo_action_embeddings2 = self.embed_oppo_action2(oppo_actions[2])

        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        my_idx_input = torch.ones(batch_size, seq_length, dtype=torch.long, device=obs.device) * self.my_idxs[0]
        my_idx_embeddings = self.embed_agent_idx(my_idx_input)
        
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings + my_idx_embeddings
        action_embeddings = action_embeddings + time_embeddings + my_idx_embeddings
        
        if self.args.env_type in ["oc", "lbf"]:
            oppo_idx_input = torch.ones(batch_size, seq_length, dtype=torch.long, device=obs.device) * self.oppo_idxs[0]
            oppo_idx_embeddings = self.embed_agent_idx(oppo_idx_input)
            oppo_action_embeddings = oppo_action_embeddings + time_embeddings + oppo_idx_embeddings
        elif self.args.env_type == "pp":
            oppo_idx_input0 = torch.ones(batch_size, seq_length, dtype=torch.long, device=obs.device) * self.oppo_idxs[self.argsorted_oppo_idxs[0]]
            oppo_idx_embeddings0 = self.embed_agent_idx(oppo_idx_input0)
            oppo_action_embeddings0 = oppo_action_embeddings0 + time_embeddings + oppo_idx_embeddings0
            oppo_idx_input1 = torch.ones(batch_size, seq_length, dtype=torch.long, device=obs.device) * self.oppo_idxs[self.argsorted_oppo_idxs[1]]
            oppo_idx_embeddings1 = self.embed_agent_idx(oppo_idx_input1)
            oppo_action_embeddings1 = oppo_action_embeddings1 + time_embeddings + oppo_idx_embeddings1
            oppo_idx_input2 = torch.ones(batch_size, seq_length, dtype=torch.long, device=obs.device) * self.oppo_idxs[self.argsorted_oppo_idxs[2]]
            oppo_idx_embeddings2 = self.embed_agent_idx(oppo_idx_input2)
            oppo_action_embeddings2 = oppo_action_embeddings2 + time_embeddings + oppo_idx_embeddings2
        elif self.args.env_type == "Harfang":
            oppo_idx_input = torch.ones(batch_size, seq_length, dtype=torch.long, device=obs.device) * self.oppo_idxs[0]
            oppo_idx_embeddings = self.embed_agent_idx(oppo_idx_input)
            oppo_action_embeddings = oppo_action_embeddings + time_embeddings + oppo_idx_embeddings
            
        returns_embeddings = returns_embeddings + time_embeddings
        
        if self.args.env_type in ["oc", "lbf"]:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, oppo_action_embeddings, returns_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, (3+len(self.oppo_idxs))*seq_length, self.hidden_size)  
        elif self.args.env_type == "pp":
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, oppo_action_embeddings0, oppo_action_embeddings1, oppo_action_embeddings2, returns_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, (3+len(self.oppo_idxs))*seq_length, self.hidden_size)
        elif self.args.env_type == "Harfang":
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, oppo_action_embeddings, returns_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, (3+len(self.oppo_idxs))*seq_length, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, *[attention_mask for _ in range(len(self.oppo_idxs))], attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, (3+len(self.oppo_idxs))*seq_length)
        
        if self.args.env_type in ["oc", "lbf", "Harfang"]:
            prompt_seq_length = prompt_states[0].shape[1]
            if self.args.env_type == "oc":
                prompt_states_ = prompt_states[0].permute(0, 1, 4, 2, 3).reshape(-1, self.oppo_obs_dim[-1], self.oppo_obs_dim[0], self.oppo_obs_dim[1])
            else:
                prompt_states_ = prompt_states[0]
            prompt_state_embeddings = self.prompt_embed_state(prompt_states_).reshape(batch_size, prompt_seq_length, self.hidden_size)
            prompt_action_embeddings = self.prompt_embed_action(prompt_actions[0])
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps[0])
            idx_input = torch.ones(batch_size, prompt_seq_length, dtype=torch.long, device=prompt_states[0].device) * self.oppo_idxs[0]
            prompt_idx_embeddings = self.embed_agent_idx(idx_input)

            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings + prompt_idx_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings + prompt_idx_embeddings

            prompt_stacked_inputs = torch.stack(
                (prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * prompt_seq_length, self.hidden_size)

            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask[0], prompt_attention_mask[0]), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2 * prompt_seq_length)
        elif self.args.env_type == "pp":
            prompt_seq_length0 = prompt_states[0].shape[1]
            prompt_state_embeddings0 = self.prompt_embed_state0(prompt_states[0]).reshape(batch_size, prompt_seq_length0, self.hidden_size)
            prompt_action_embeddings0 = self.prompt_embed_action0(prompt_actions[0])
            prompt_time_embeddings0 = self.prompt_embed_timestep(prompt_timesteps[0])
            idx_input0 = torch.ones(batch_size, prompt_seq_length0, dtype=torch.long, device=prompt_states[0].device) * self.oppo_idxs[self.argsorted_oppo_idxs[0]]
            prompt_idx_embeddings0 = self.embed_agent_idx(idx_input0)

            prompt_state_embeddings0 = prompt_state_embeddings0 + prompt_time_embeddings0 + prompt_idx_embeddings0
            prompt_action_embeddings0 = prompt_action_embeddings0 + prompt_time_embeddings0 + prompt_idx_embeddings0
            
            prompt_stacked_inputs0 = torch.stack(
                (prompt_state_embeddings0, prompt_action_embeddings0), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * prompt_seq_length0, self.hidden_size)

            prompt_stacked_attention_mask0 = torch.stack(
                (prompt_attention_mask[0], prompt_attention_mask[0]), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2 * prompt_seq_length0)
            
            prompt_seq_length1 = prompt_states[1].shape[1]
            prompt_state_embeddings1 = self.prompt_embed_state1(prompt_states[1]).reshape(batch_size, prompt_seq_length1, self.hidden_size)
            prompt_action_embeddings1 = self.prompt_embed_action1(prompt_actions[1])
            prompt_time_embeddings1 = self.prompt_embed_timestep(prompt_timesteps[1])
            idx_input1 = torch.ones(batch_size, prompt_seq_length1, dtype=torch.long, device=prompt_states[1].device) * self.oppo_idxs[self.argsorted_oppo_idxs[1]]
            prompt_idx_embeddings1 = self.embed_agent_idx(idx_input1)
            
            prompt_state_embeddings1 = prompt_state_embeddings1 + prompt_time_embeddings1 + prompt_idx_embeddings1
            prompt_action_embeddings1 = prompt_action_embeddings1 + prompt_time_embeddings1 + prompt_idx_embeddings1
            
            prompt_stacked_inputs1 = torch.stack(
                (prompt_state_embeddings1, prompt_action_embeddings1), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * prompt_seq_length1, self.hidden_size)
            
            prompt_stacked_attention_mask1 = torch.stack(
                (prompt_attention_mask[1], prompt_attention_mask[1]), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2 * prompt_seq_length1)
            
            prompt_seq_length2 = prompt_states[2].shape[1]
            prompt_state_embeddings2 = self.prompt_embed_state2(prompt_states[2]).reshape(batch_size, prompt_seq_length2, self.hidden_size)
            prompt_action_embeddings2 = self.prompt_embed_action2(prompt_actions[2])
            prompt_time_embeddings2 = self.prompt_embed_timestep(prompt_timesteps[2])
            idx_input2 = torch.ones(batch_size, prompt_seq_length2, dtype=torch.long, device=prompt_states[2].device) * self.oppo_idxs[self.argsorted_oppo_idxs[2]]
            prompt_idx_embeddings2 = self.embed_agent_idx(idx_input2)
            
            prompt_state_embeddings2 = prompt_state_embeddings2 + prompt_time_embeddings2 + prompt_idx_embeddings2
            prompt_action_embeddings2 = prompt_action_embeddings2 + prompt_time_embeddings2 + prompt_idx_embeddings2
            
            prompt_stacked_inputs2 = torch.stack(
                (prompt_state_embeddings2, prompt_action_embeddings2), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * prompt_seq_length2, self.hidden_size)
            
            prompt_stacked_attention_mask2 = torch.stack(
                (prompt_attention_mask[2], prompt_attention_mask[2]), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2 * prompt_seq_length2)

        # stacked_inputs add prompted sequence
        if self.args.env_type in ["oc", "lbf"]:
            stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
        elif self.args.env_type == "pp":
            stacked_inputs = torch.cat((prompt_stacked_inputs0, prompt_stacked_inputs1, prompt_stacked_inputs2, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask0, prompt_stacked_attention_mask1, prompt_stacked_attention_mask2, stacked_attention_mask), dim=1)
        elif self.args.env_type == "Harfang":
            stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
            stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.gpt_decoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        x = x[:, -((3+len(self.oppo_idxs))*seq_length):, :]
        x = x.reshape(batch_size, seq_length, (3+len(self.oppo_idxs)), self.hidden_size).permute(0, 2, 1, 3)

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        value_preds = self.predict_value(x[:,0])[:, -seq_length:, :]  # predict value given state V(s)
        action_preds = self.predict_action(x[:,0])[:, -seq_length:, :]  # predict next action given state pi(a|s)
        
        if self.args.env_type in ["oc", "lbf"]:
            oppo_action_preds = self.predict_oppo_action(x[:,0])[:, -seq_length:, :]
            oppo_action_preds = torch.stack([oppo_action_preds], dim=0).to(dtype=torch.float32, device=action_preds.device)
        elif self.args.env_type == "pp":
            oppo_action_preds0 = self.predict_oppo_action0(x[:,0])[:, -seq_length:, :]
            oppo_action_preds1 = self.predict_oppo_action1(x[:,0])[:, -seq_length:, :]
            oppo_action_preds2 = self.predict_oppo_action2(x[:,0])[:, -seq_length:, :]
            oppo_action_preds = torch.stack([oppo_action_preds0, oppo_action_preds1, oppo_action_preds2], dim=0).to(dtype=torch.float32, device=action_preds.device)
        elif self.args.env_type == "Harfang":
            oppo_action_preds = self.predict_oppo_action(x[:,0])[:, -seq_length:, :]
            oppo_action_preds = torch.stack([oppo_action_preds], dim=0).to(dtype=torch.float32, device=action_preds.device)
        
        # pi(a|s), V(s), pi(a_oppo|s)
        return action_preds, value_preds, oppo_action_preds
    
    def load_model(self, param_path, device="cpu"):
        self.load_state_dict(
            torch.load(param_path, map_location=torch.device(device))
        )