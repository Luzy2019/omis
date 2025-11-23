#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np
import cv2

class Policy(object):
    def __init__(self, env, agent_index, num_step=100):
        self.env = env
        self.agent_index = agent_index
        self.num_step = num_step
        self.cur_step = 0
    
    def rollout_one_step(self):
        u_other = np.zeros(self.env.world.dim_p)
        p_force_ = [u_other for _ in range(len(self.env.world.entities))]
        # we rollout each action for 1 step to see the next observation
        all_obs = []
        for i in range(5):
            u_ = np.zeros(self.env.world.dim_p)
            if i == 1: u_[0] = -1.0
            if i == 2: u_[0] = 1.0
            if i == 3: u_[1] = -1.0
            if i == 4: u_[1] = 1.0
            u_ *= 5.0
            p_force_[self.agent_index] = u_
            p_force_ = self.env.world.apply_environment_force(p_force_)
            p_vel_ = self.env.world.agents[self.agent_index].state.p_vel * (1 - self.env.world.damping)
            p_vel_ += (p_force_[self.agent_index] / self.env.world.agents[self.agent_index].mass) * self.env.world.dt
            p_pos_ = self.env.world.agents[self.agent_index].state.p_pos + p_vel_ * self.env.world.dt
            
            entity_pos_ = []
            for entity in self.env.world.landmarks:
                entity_pos_.append(entity.state.p_pos - p_pos_)
            other_pos_ = []
            for other in self.env.world.agents:
                if other is self.env.world.agents[self.agent_index]: continue
                other_pos_.append(other.state.p_pos - p_pos_)
            obs_i = np.concatenate(entity_pos_ + other_pos_)
            all_obs.append(obs_i)
        
        return all_obs
    
    def reset(self):
        raise NotImplementedError()
    
    def action(self, obs):
        raise NotImplementedError()


class RandomPolicy(Policy):
    # NOTE: only used for testing the environment
    def __init__(self, env, agent_index, num_step=100):
        super(RandomPolicy, self).__init__(env, agent_index, num_step)
    def action(self, obs):
        a = np.random.randint(0,5)
        if self.env.discrete_action_input:
            u = np.copy(a)
        else:
            u = np.zeros(5) # 5-d because of no-move action
            u[a] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])


def gen_video(imgs, filename, size=(1000, 1000)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, 5.0, size)
    for i in imgs:
        Img = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
        Img = cv2.resize(Img, size)
        video_writer.write(Img)
    print(f"Video {filename} writen down!")

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-sc', '--scenario', default='simple_tag.py', help='Path of the scenario Python script.')
    parser.add_argument('-pp', '--steps', default=100, help='Number of steps for testing episode')
    args = parser.parse_args()
    num_steps = args.steps
    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    imgs = []
    img_size = (2000,2000)
    img_scale = (0.2,0.2)
    view_index = 0
    img = np.array(env.render(mode='rgb_array',size=img_size,scale=img_scale))
    # print(img.shape)
    imgs.append(img[view_index])
    # create interactive policies for each agent
    # test_policy = FixOnePolicy
    # test_policy = ChaseOnePolicy
    # test_policy = BouncePolicy
    # test_policy = MiddlePolicy
    # test_policy = FixThreePolicy
    test_policy = RandomPolicy
    # policies += [RLPolicy(env, i, 'checkpoints/RL/rl_params_ChaseOnePolicy_v0_10000.pt') for i in range(1, env.n)]
    policies = [test_policy(env, i, num_steps) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    returns = [0. for _ in range(env.n)]
    for i in range(num_steps):
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        img = np.array(env.render(mode='rgb_array',size=img_size,scale=img_scale))
        # print(img.shape)
        imgs.append(img[view_index])
        # display rewards
        for k in range(env.n):
            returns[k] += reward_n[k]
    for k, agent in enumerate(env.world.agents):
        print(agent.name + " return: %0.3f" % returns[k])
    if not os.path.exists("videos"):
        os.makedirs("videos")
    filename = f"videos/{args.scenario.split('.')[0]}_{test_policy.__name__}.mp4"
    gen_video(imgs, filename, img_size)