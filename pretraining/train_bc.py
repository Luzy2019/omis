import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from bc import Agent as BCAgent
from myutils.plot import plot_3d_trajectories, plot_distance
from myutils.data_processor import read_data
from myutils.buffer import *
from myutils.seed import *

from envs.HarfangEnv_GYM.HarfangEnv_GYM import *
import envs.HarfangEnv_GYM.dogfight_client as df

import numpy as np
import time
import math
from statistics import mean, pstdev
import datetime

import csv
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

def validate(validationEpisodes, env:HarfangEnv, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, tensor_writer:SummaryWriter, highScore, successRate, if_random):          
    success = 0
    fire_success = 0
    valScores = []
    self_pos = []
    oppo_pos = []
    for e in range(validationEpisodes):
        distance=[]
        fire=[]
        lock=[]
        missile=[]
        if if_random: state = env.random_reset()
        else: state = env.reset()
        totalReward = 0
        done = False
        for step in range(validationStep):
            if not done:
                action = agent.chooseActionNoNoise(state)
                n_state, reward, done, info, iffire, beforeaction, afteraction, locked, step_success = env.step_test(action)
                state = n_state
                totalReward += reward
                
                distance.append(env.loc_diff)
                if iffire:
                    fire.append(step)
                if locked:
                    lock.append(step)
                if beforeaction:
                    missile.append(step)
                
                if e == validationEpisodes - 1:
                    self_pos.append(env.get_pos())
                    oppo_pos.append(env.get_oppo_pos())

                if step == validationStep - 1:
                    break

            elif done:
                if env.episode_success:
                    success += 1
                if env.fire_success:
                    fire_success += 1
                break

        valScores.append(totalReward)
        
    if mean(valScores) > highScore or success/validationEpisodes >= successRate or arttir % 5 == 0:
        agent.saveCheckpoints("Agent{}_{}_{}_".format(arttir, round(success/validationEpisodes*100), round(mean(valScores))), model_dir)
        if plot:
            plot_3d_trajectories(self_pos, oppo_pos, fire, lock, plot_dir, f'trajectories_{arttir}.png') 
            plot_distance(distance, lock, missile, fire, plot_dir, f'distance_{arttir}.png')
            
            os.makedirs(plot_dir+'/csv', exist_ok=True)
            with open(plot_dir+'/csv/self_pos{}.csv'.format(arttir), 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(self_pos)
            with open(plot_dir+'/csv/oppo_pos{}.csv'.format(arttir), 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(oppo_pos)
            with open(plot_dir+'/csv/fire{}.csv'.format(arttir), 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows([[item] for item in fire])
            with open(plot_dir+'/csv/lock{}.csv'.format(arttir), 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows([[item] for item in lock])
            with open(plot_dir+'/csv/distance{}.csv'.format(arttir), 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows([[item] for item in distance])

        if mean(valScores) > highScore: # 总奖励分数
            highScore = mean(valScores)
        if success / validationEpisodes >= successRate: # 追逐成功率
            successRate = success / validationEpisodes

    print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes, ' Fire Success Rate:', fire_success/validationEpisodes)
    tensor_writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
    tensor_writer.add_scalar('Validation/Std Reward', pstdev(valScores), episode)
    tensor_writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
    tensor_writer.add_scalar('Validation/Fire Success Rate', fire_success/validationEpisodes, episode)

    return highScore, successRate

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

def main(config):
    print('gpu is ' + str(torch.cuda.is_available()))

    agent_name = config.agent
    # port = config.port
    # hirl_type = config.type
    # bc_weight = config.bc_weight
    model_name = config.model_name
    load_model = config.load_model
    render = not (config.render)
    plot = config.plot
    env_type = config.env
    
    if config.random:
        print("random")
    else:
        print("fixed")
    if_random = config.random

    if config.seed is not None:
        set_seed(config.seed)
        print(f"successful set seed: {config.seed}")
    else:
        print("no seed is set")

    if not render:
        print('rendering mode')
    else:
        print('no rendering mode')

    with open('local_config.yaml', 'r') as file:
        local_config = yaml.safe_load(file)

    if local_config['network']['ip'] == 'YOUR_IP_ADDRESS':
        raise ValueError("Please update the 'network.ip' field in config.yaml with your own IP address.")

    # df.connect(local_config["network"]["ip"], port)

    start = time.time() #STARTING TIME
    # df.disable_log()

    name = "Harfang_GYM"

    # PARAMETERS
    if env_type == "straight_line":
        print("env is harfang straight line")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 1500 # 6000
        validationStep = 1500 # 6000
        
        # env = HarfangEnv()

    elif env_type == "serpentine":
        print("env is harfang serpentine")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 1500 # 6000
        validationStep = 1500 # 6000
            
        # env = HarfangSerpentineEnv()

    elif env_type == "circular":
        print("env is harfang circular")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 1900 # 6000
        validationStep = 1900 # 6000
            
        # env = HarfangCircularEnv()

    elif env_type == "low_blood":
        print("env is harfang low blood")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 3800 # 6000
        validationStep = 3800 # 6000
            
        # env = HarfangLowBloodEnvNew()

    elif env_type == "normal_blood":
        print("env is harfang normal blood")
        trainingEpisodes = 6000
        validationEpisodes = 50 # 20
        explorationEpisodes = 20 # 200
        maxStep = 3800 # 6000
        validationStep = 3800 # 6000
            
        # env = HarfangNormalBloodEnvNew()

    else:
        print("env is not supported")
        exit()

    # df.set_renderless_mode(render)
    # df.set_client_update_mode(True)

    bufferSize = 10**5 # 10**6
    gamma = 0.99
    criticLR = 1e-4
    actorLR = 1e-4
    tau = 0.005
    checkpointRate = 25 # 25
    logRate = 300
    highScore = -math.inf
    successRate = 0
    batchSize = 16 # 128
    hiddenLayer1 = 64
    hiddenLayer2 = 128
    stateDim = 13
    actionDim = 4
    useLayerNorm = True
    # expert_warm_up = True
    # warm_up_rate = 10
    # bc_warm_up = False
    # bc_warm_up_weight = 0 # 不能动

    if if_random: data_dir = f'expert_data/{env_type}/expert_data_ai_random.csv'
    elif not if_random: data_dir = f'expert_data/{env_type}/expert_data_ai.csv'

    start_time = datetime.datetime.now()
    log_dir = local_config["experiment"]["result_dir"] + "/" + env_type + "/" + agent_name + "/" + model_name + "/" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute)
    model_dir = os.path.join(log_dir, 'model')
    summary_dir = os.path.join(log_dir, 'summary')
    plot_dir = os.path.join(log_dir, 'plot')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if agent_name == 'BC':
        expert_states, expert_actions = read_data(data_dir)
        agent = BCAgent(actorLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, useLayerNorm, name, batchSize, expert_states, expert_actions)

    # save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validationStep=validationStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2,agent=agent,model_dir=model_dir,hirl_type=hirl_type, data_dir=data_dir)
    # env.save_parameters_to_txt(log_dir)

    writer = SummaryWriter(summary_dir)
    
    arttir = 1
    if load_model:
        agent.loadCheckpoints(f"Agent20_successRate0.64", model_dir)

    # only bc
    for episode in range(trainingEpisodes):
        for step in range(maxStep):
            bc_loss = agent.train_actor()
            if step == 1000:
                writer.add_scalar('Loss/BC_Loss', bc_loss, step + episode * maxStep)
        now = time.time()
        seconds = int((now - start) % 60)
        minutes = int(((now - start) // 60) % 60)
        hours = int((now - start) // 3600)
        print('Episode: ', episode+1, 'RunTime: ', hours, ':',minutes,':', seconds)

        # validation
        # if ((episode + 1) % checkpointRate == 0 and (episode + 1) >= 1000):
        #     highScore, successRate = validate(validationEpisodes, env, validationStep, agent, plot, plot_dir, arttir, model_dir, episode, checkpointRate, writer, highScore, successRate, if_random)
        #     arttir += 1
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='BC', 
                        help="Specify the agent to use: 'HIRL', 'BC' or 'TD3'. Default is 'HIRL'.")
    # parser.add_argument('--port', type=int, default=None,
    #                     help="The port number for the training environment. Example: 12345.")
    # parser.add_argument('--type', type=str, default='soft',
    #                     help="Type of HIRL algorithm to use: 'linear', 'fixed', or 'soft'. Default is 'soft'.")
    # parser.add_argument('--bc_weight', type=float, default=0.5,
    #                     help="Weight for the behavior cloning (BC) loss. Default is 0.5.")
    parser.add_argument('--model_name', type=str, default='BC',
                        help="Name of the model to be saved. Example: 'HIRL_soft'.")
    parser.add_argument('--load_model', action='store_true',
                        help="Flag to load a pre-trained model. Use this if you want to resume training.")
    parser.add_argument('--render', action='store_true',
                        help="Flag to enable rendering of the environment.")
    parser.add_argument('--plot', action='store_true',
                        help="Flag to plot training metrics. Use this for visualization.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility. Default is None (no seed).")
    parser.add_argument('--env', type=str, default='straight_line',
                        help="Specify the training environment type: 'straight_line', 'serpentine', 'circular', 'low_blood' or 'normal_blood'. Default is 'straight_line'.")
    parser.add_argument('--random', action='store_true', default=True,
                        help="Flag to use random initialization in the environment. Default is True (random).")
    main(parser.parse_args())

    # python pretraining/train_bc.py --env serpentine --random --plot