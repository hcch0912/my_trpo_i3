#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
# from two_player.pong import PongGame
import sys
sys.path.append('../../')
class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

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
def get_trainers(env, num_adversaries, obs_dim, act_dim, arglist):
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
    trainers = []
    loggers = []
    trainer = Policy
    for i in range(num_adversaries):
        trainers.append( Policy(obs_dim, act_dim, arglist.kl_targ, arglist.hid1_mult, arglist.policy_logvar))
        loggers.append(Logger(logname='player_{}'.format(i), now = now))
    for i in range(num_adversaries, env.n):
        trainers.append( Policy(obs_dim, act_dim, arglist.kl_targ, arglist.hid1_mult, arglist.policy_logvar))
        loggers.append(Logger(logname='player_{}'.format(i), now = now))
    return trainers, loggers



def run_episode(env, trainers, scaler,max_len, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    obs_record, act_record, rew_record = [[] for i in range(len(obs))], [[] for i in range(len(obs))],[[] for i in range(len(obs))]

    done = [False for i in range(len(obs))]
    step = 0.0
    count =1

    # scale, offset = scaler.get()
    # scale[-1] = 1.0  # don't scale time step feature
    # offset[-1] = 0.0  # don't offset time step feature
    # print(all(done))

    while not all(done) and count <= max_len:
        if animate:
            env.render('human')
        for i in range(len(obs)):
            obs_this = np.array(obs[i]).reshape((1,-1))
            # print("---------")
            # print(np.array(obs_this).shape)
            obs_this = np.append(obs_this, [[step]], axis = 1)
            # print(np.array(obs_this).shape)
            action = trainers[i].sample(obs_this).reshape((1,-1)).astype(np.float32)
            obs_record[i].append(np.reshape(obs_this, (-1)))
            act_record[i].append(np.reshape(action, (-1)))
        this_action = [act[-1] for act in act_record]    
        obs, reward, done, _ = env.step(this_action)  
        # print(one)  
        for i in range(len(reward)):
            rew_record[i].append(np.reshape(reward[i], (-1,)))
        step += 1e-3  # increment time step feature
        count +=1
        if True in done or 1 in done:
            obs = env.reset()    
    # print("finish")        
    # print(np.array(obs_record).shape)
    return (obs_record, act_record, rew_record)


def run_policy(env, policys, scaler, loggers, max_len,episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = [[] for i in range(len(policys))]
    for e in range(episodes):
        observations, actions, rewards = run_episode(env, policys, scaler, max_len)
        # print(len(observations[0][0][0]))
        total_steps += len(observations[0])
        for i in range(len(observations)):
            trajectories[i].append(
                {'observes': observations[i],
                      'actions': actions[i],
                      'rewards': np.array(rewards[i]),
                      'values': None
                      }
                )

            loggers[i].log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories[i]]),
                'Steps': total_steps})

    # scaler.update(unscaled)  # update running statistics for scaling observations
    # print(np.array(trajectories).shape)
    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for agent_trajectory in trajectories:
        for trajectory in agent_trajectory:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            disc_sum_rew = discount(rewards, gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    # print("---------")
    # print(len(trajectories), print(len(trajectories[0])))
    for agent_trajectories in trajectories:
        for trajectory in agent_trajectories:
            observes = trajectory['observes']
            # print("---herer")
            # print(np.arra(observes).shape)
            # print(np.array(observes))
            values = val_func.predict(observes)

            # print(values)
            trajectory['values'] = values
            # print(trajectory['values'])


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for agent_trajectory in trajectories:
        for trajectory in agent_trajectory:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            # print(values)
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = discount(tds, gamma * lam)
            trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = [[] for i in range(len(trajectories))]
    actions = [[] for i in range(len(trajectories))]
    advantages = [[] for i in range(len(trajectories))]
    disc_sum_rew = [[] for i in range(len(trajectories))]

    for i  in range(len(trajectories)):
        observes[i] = np.concatenate([t['observes'] for t in trajectories[i]])
        actions[i] = np.concatenate([t['actions'] for t in trajectories[i]])
        disc_sum_rew[i] = np.concatenate([t['disc_sum_rew'] for t in trajectories[i]])
        disc_sum_rew[i] = np.reshape(disc_sum_rew[i], (-1,))
        advantages[i] = np.concatenate([t['advantages'] for t in trajectories[i]])
        # normalize advantages

        advantages[i] = (advantages[i] - advantages[i].mean()) / (advantages[i].std() + 1e-6)
        advantages[i] = np.array([ a.mean() for a in advantages[i]])
        # print(advantages[i].shape)
        # print(disc_fsum_rew[i].shape)
    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, loggers, episode):
    """ Log various batch statistics """
    for i in range(len(loggers)):

        loggers[i].log({'_mean_obs': np.mean(observes[i]),
                    '_min_obs': np.min(observes[i]),
                    '_max_obs': np.max(observes[i]),
                    '_std_obs': np.mean(np.var(observes[i], axis=0)),
                    '_mean_act': np.mean(actions[i]),
                    '_min_act': np.min(actions[i]),
                    '_max_act': np.max(actions[i]),
                    '_std_act': np.mean(np.var(actions[i], axis=0)),
                    '_mean_adv': np.mean(advantages[i]),
                    '_min_adv': np.min(advantages[i]),
                    '_max_adv': np.max(advantages[i]),
                    '_std_adv': np.var(advantages[i]),
                    '_mean_discrew': np.mean(disc_sum_rew[i]),
                    '_min_discrew': np.min(disc_sum_rew[i]),
                    '_max_discrew': np.max(disc_sum_rew[i]),
                    '_std_discrew': np.var(disc_sum_rew[i]),
                    '_Episode': episode
                    })


def main(arglist):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    killer = GracefulKiller()
    # env, obs_dim, act_dim = init_gym(aenv_name)
    env = make_env(arglist.scenario, arglist)
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].n
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    aigym_path = os.path.join('/tmp', arglist.scenario, now)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, arglist.hid1_mult)
    trainers, loggers = get_trainers(env, arglist.num_adversaries, obs_dim, act_dim, arglist)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, trainers, scaler, loggers, arglist.max_episode_len , episodes=5)
    episode = 0
    while episode < arglist.num_episodes:
        trajectories = run_policy(env, trainers, scaler, loggers, arglist.max_episode_len ,  episodes=arglist.b_size)
        episode += len(trajectories[0])
        print("episode: {}".format(episode))
        add_value(trajectories, val_func)
        add_disc_sum_rew(trajectories, arglist.gamma)
        add_gae(trajectories, arglist.gamma, arglist.lam)
        observations, actions, advantages, disc_sum_rews = build_train_set(trajectories)
        log_batch_stats(observations, actions, advantages, disc_sum_rews, loggers, episode)
        for i in range(len(trainers)):
            trainers[i].update(observations[i], actions[i], advantages[i], loggers[i])
            val_func.fit(observations[i], disc_sum_rews[i], loggers[i])  
            loggers[i].write(display=True)  

        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
        
            
        # score = play(env, policy1, policy2)   
    for i in range(len(loggers)):
        loggers[i].close()
        trainers[i].close_sess()
        val_func.close_sess()     



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('--scenario', type=str, help='OpenAI Gym environment name')
    parser.add_argument('--n',  type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--b_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    # parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
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

    arglist = parser.parse_args()
    main(arglist)
