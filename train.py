import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import I3MADDPGAgentTrainer
from origin_maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

import collections
from copy import deepcopy
import random

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="my_model", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    #extra 
    parser.add_argument("--timestep", type = int, default = 5)
    parser.add_argument("--seed", type = int, default = 10)
    parser.add_argument("--good-i3", type = int, default = 1)
    parser.add_argument("--adv-i3", type = int, default = 1)
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from Env.multiagent.environment import MultiAgentEnv
    import Env.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    I3trainer = I3MADDPGAgentTrainer
    Orgintrainer = MADDPGAgentTrainer
    act_traj_space = [((env.n-1),  arglist.timestep ,  env.action_space[0].n) for i in range(env.n)] 
    intent_shape = [((env.n-1) * env.action_space[0].n, ) for i in range(env.n)]
    # with tf.device("/device:GPU:0"):
    print(arglist.adv_i3, arglist.good_i3)
    if arglist.adv_i3 == 1:
        print("i3 adv")
        for i in range(num_adversaries):
            trainers.append(I3trainer(
                    "agent_%d" % i, model, obs_shape_n, env.action_space, act_traj_space,intent_shape, i, arglist,
                    local_q_func=(arglist.adv_policy=='ddpg')))
    else:
        for i in range(num_adversaries):
            trainers.append(Orgintrainer(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')
                ))        
    if arglist.good_i3 == 1:   
        print("i3 good")     
        for i in range(num_adversaries, env.n):
            trainers.append(I3trainer(
                    "agent_%d" % i, model, obs_shape_n, env.action_space,act_traj_space, intent_shape, i, arglist,
                    local_q_func=(arglist.good_policy=='ddpg')))
    else:
        for i in range(num_adversaries, env.n):
            trainers.append(Orgintrainer(
                "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                local_q_func=(arglist.good_policy=='ddpg')))        
    return trainers

def get_traj_n(act_trajs):
    act_traj = []

    for i in range(len(act_trajs)):
        act_traj.append([])
        for j in range(len(act_trajs)):
            if i != j:
     
                a = deepcopy(act_trajs[j])
                act_traj[i].append(a)
 
    return np.array(act_traj)


def train(arglist):
    
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []
        final_ep_accurancy =[]  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        record_accurancy = [[]] * env.n
        #initialize act trajectories
        act_trajs = []
        for i in range(env.n):
            act_trajs.append(collections.deque(np.zeros((arglist.timestep, env.action_space[0].n)), maxlen = arglist.timestep) )
      
        print('Starting iterations...')
        while True:
            # get action

            act_traj_n = get_traj_n(act_trajs)
            
            if arglist.adv_i3 == 1 and arglist.good_i3 == 1:
                intent_n = [agent.intent(obs, act_traj) for agent, obs, act_traj in zip(trainers, obs_n, act_traj_n)]
                action_n = [agent.action(obs, intent) for agent, obs,intent in zip(trainers,obs_n, intent_n)]
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                # collect experience

                for i in range(len(act_trajs)):
                    act_trajs[i].append(action_n[i])

                act_traj_next_n = get_traj_n(act_trajs)
                intent_next_n = [agent.intent(obs, act_traj) for agent, obs, act_traj in zip(trainers, new_obs_n, act_traj_next_n)]

                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], act_traj_n[i], intent_n[i],act_traj_next_n[i], intent_next_n[i], done_n[i], terminal)
            elif arglist.adv_i3 == 1 and arglist.good_i3 == 0:
                #adv use I3 good use maddpg
                intent_n = []
                action_n = []
                for i in range(len(trainers)):
                    if i < arglist.num_adversaries:
                        intent = trainers[i].intent(obs_n[i], act_traj_n[i])
                        action = trainers[i].action(obs_n[i], intent)
                        action_n.append(action)
                        intent_n.append(intent)
                    else:
                        action = trainers[i].action(obs_n[i])    
                        action_n.append(action)
                        intent_n.append(np.zeros((np.array(intent).shape)))

                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)

                for i in range(len(act_trajs)):
                    act_trajs[i].append(action_n[i])

                act_traj_next_n = get_traj_n(act_trajs)
                intent_next_n = []
                for i in range(len(trainers)):
                    if i < arglist.num_adversaries:
                        intent_next_n.append(trainers[i].intent(new_obs_n[i], act_traj_next_n[i]))
                    else:
                        intent_next_n.append(np.zeros((arglist.timestep *  (env.action_space[0].n-1))))  
                
                for i in range(len(trainers)):
                    if i < arglist.num_adversaries:
                        trainers[i].experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], act_traj_n[i], intent_n[i],act_traj_next_n[i], intent_next_n[i], done_n[i], terminal)
                    else:
                        trainers[i].experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)    
            elif arglist.good_i3 == 1 and arglist.adv_i3 ==0:
                #adv use I3 good use maddpg
                intent_n = []
                action_n = []
                for i in range(len(trainers)):
                    if i >=arglist.num_adversaries:
                        intent = trainers[i].intent(obs_n[i], act_traj_n[i])
                        action = trainers[i].action(obs_n[i], intent)
                        action_n.append(action)
                        intent_n.append(intent)
                    else:
                        action = trainers[i].action(obs_n[i])    
                        action_n.append(action)
                        intent_n.append(np.zeros((arglist.timestep *  (env.action_space[0].n-1))))

                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)

                for i in range(len(act_trajs)):
                    act_trajs[i].append(action_n[i])

                act_traj_next_n = get_traj_n(act_trajs)
                intent_next_n = []
                for i in range(len(trainers)):
                    if i  >= arglist.num_adversaries:
                        intent_next_n.append(trainers[i].intent(new_obs_n[i], act_traj_next_n[i]))
                    else:
                        intent_next_n.append(np.zeros((arglist.timestep *  (env.action_space[0].n-1))))
                
                for i in range(len(trainers)):
                    if i  >= arglist.num_adversaries:
                        trainers[i].experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], act_traj_n[i], intent_n[i],act_traj_next_n[i], intent_next_n[i], done_n[i], terminal)
                    else:
                        trainers[i].experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)    
            else:
                action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)] 
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                # collect experience
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

            obs_n = new_obs_n
          
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.5)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                if loss:   
                    record_accurancy.append(loss[2])
            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
            # if (len(record_accurancy) % arglist.save_rate == 0):
            #     print("-----------------------------")
            #     final_ep_accurancy.append(np.mean(record_accurancy[-arglist.save_rate]))
            #     print(final_ep_accurancy)    
            if (len(record_accurancy) % 100 == 0):
                final_ep_accurancy.append(np.mean(record_accurancy[-100]))     

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + str(arglist.seed) + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + str(arglist.seed) + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                acc_file = arglist.plots_dir + arglist.exp_name + str(arglist.seed) + '_accurancy.pkl'    
                with open(acc_file, 'wb') as fp:
                    pickle.dump(final_ep_accurancy, fp)    
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':

    arglist = parse_args()
    random.seed(arglist.seed)
    train(arglist)
