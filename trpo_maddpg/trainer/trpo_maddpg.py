import numpy as np
import random
import tensorflow as tf
import trpo_maddpg.common.tf_util as U

from trpo_maddpg.common.distributions import make_pdtype
from trpo_maddpg import AgentTrainer
from trpo_maddpg.trainer.replay_buffer import ReplayBuffer

from copy import deepcopy

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]
def get_true_action(act_traj_next_n):

            true_actions = []
            for i in range(len(act_traj_next_n)):
                true_actions.append([])
                agent = act_traj_next_n[i]
                for j in range(len(agent)):
                    true_actions[i].append([])
                    for k in range(len(agent[j])):
                        a = deepcopy(agent[j][k][-1])
                        true_actions[i][j] = np.concatenate((true_actions[i][j],a), axis = 0)
            return  true_actions
def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def trpo_loss(logvars, act_ph_n, old_log_vars_ph, old_i_ph,advantage_ph, output_size, beta, eta, kl_targ):
    #the advantage here is the critic value 
    logp = -0.5 * tf.reduce_sum(logvars)
    logp += 0.5 * tf.reduce_sum(tf.square(act_ph_n - i)/ tf.exp(logvars), axis =1)
    logp_old = -0.5 * tf.reduce_sum(old_log_vars_ph)
    logp_old += -0.5 * tf.reduce_sum(tf.square(act_ph_n - old_i_ph))
        #TRPO kl entropy
    log_det_cov_old = tf.reduce_sum(old_log_vars_ph)
    log_det_cov_new = tf.reduce_sum(logvars)
    tr_old_new = tf.reduce_sum(tf.exp(old_log_vars_ph - logvars))
    kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
            tf.reduce_sum(tf.square(i - old_i_ph) / tf.exp(logvars), axis =1) - 
            output_size)
    entropy = 0.5 *(output_size * (np.log(2 * np.pi) +1) + tf.reduce_sum(logvars))

        #TRPO loss
    loss1 = -tf.reduce_mean(advantage_ph * tf.exp(logp - logp_old))
    loss2 = tf.reduce_mean(beta * kl)
    loss3 = eta * tf.square(tf.maximium(0.0, kl -2.0 * kl_targ))
    loss = loss1 + loss2 + loss3
    return loss, kl, entropy

def i_train(make_obs_ph_n, intent_ph_n, act_space_n,make_act_ph, make_intent_ph_n, make_log_vars_ph, make_old_target_ph, make_old_i_ph, make_act_traj_ph_n, 
    i_func, i_index,output_size ,  optimizer, scope, reuse, grad_norm_clipping=None, num_units=64,
    beta_ph = make_beta_ph, lr_ph = make_lr_ph, eta= 50, kl_targ = 0.003, policy_logvar = -1.0):
    with tf.variable_scope(scope, reuse=reuse):

        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        obs_ph_n = make_obs_ph_n
        #the same size
        intent_ph_n = make_intent_ph_n
        act_ph_n = make_act_ph
        old_log_vars_ph = make_log_vars_ph
        old_i_ph = make_old_i_ph
        beta = beta_ph
        lr = lr_ph
        # use the critic value from the q  
        advantage_ph = tf.placeholder(tf.float32, (None,), 'advantages')

        flat_act_traj_ph_n =[tf.reshape(a, (-1,  a.shape[1] * a.shape[2] *a.shape[3])) for a in make_act_traj_ph_n] 

        act_traj_ph_n = make_act_traj_ph_n

        i_input = [tf.concat([obs, act_traj], axis = 1) for obs, act_traj in zip(obs_ph_n, flat_act_traj_ph_n)]

        i, logvars = i_func(i_input[i_index], output_size, policy_logvar,  scope = "i_func", num_units = 64 )
        i_func_vars = U.scope_vars(U.absolute_scope_name("i_func"))

        #TRPO objectives
        loss, kl, entropy= trpo_loss(logvars, act_ph_n, old_log_vars_ph, old_i_ph, advantage_ph,output_size, beta, eta, kl_targ)

        optimize_expr = U.minimize_and_clip(optimizer, loss, i_func_vars, grad_norm_clipping)
        train = U.function(inputs= obs_ph_n + act_traj_ph_n + intent_ph_n + [lr_ph] + [beta_ph], outputs=[loss, kl, entropy], updates=[optimize_expr])
        i_values = U.function(inputs =[obs_ph_n[i_index]] + [act_traj_ph_n[i_index]], outputs = i)
        get_object = U.function(inputs= obs_ph_n + act_traj_ph_n +act_ph_n + old_log_vars_ph + old_i_ph + [advantage_ph], outputs = [kl, entropy])

        return i_values, train, get_object
        

def p_train(make_obs_ph_n, act_space_n, make_intent_ph_n, make_log_vars_ph, make_old_target_ph, make_old_i_ph,
    p_index, p_func, q_func, optimizer,policy_logvar, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None,
    beta_ph = make_beta_ph, lr_ph = make_lr_ph, eta= 50, kl_targ = 0.003, policy_logvar = -1.0):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        intent_ph_n = make_intent_ph_n
        old_log_vars_ph = make_log_vars_ph
        old_i_ph = make_old_i_ph
        beta = beta_ph
        lr = make_lr_ph
        #advantage == q_value
        advantage_ph = tf.placeholder(tf.float32, (None,), 'advantages')

        p_input = tf.concat([obs_ph_n[p_index], intent_ph_n[p_index]], axis = 1)

        p, p_logvars = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(obs_ph_n + act_input_n + intent_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index], intent_ph_n[p_index]], 1)
        q, q_logvars = q_func(q_input, 1, policy_logvar, scope="q_func", reuse=True, num_units=num_units)[:,0]
        # pg_loss = -tf.reduce_mean(q)

        p_loss, p_kl, p_entropy = trpo_loss(p_logvars, act_ph_n, old_log_vars_ph, old_i_ph, q,output_size, beta, eta, kl_targ)

        #TRPO loss. p_reg is the entropy of q distribution 
        loss = p_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + intent_ph_n + [lr_ph] + [beta_ph], outputs=[p_loss, p_kl, p_entropy], updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]] + [intent_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index], intent_ph_n[p_index]], p)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}
#value function 
def q_train(make_obs_ph_n, act_space_n,make_intent_ph_n, q_index, q_func, optimizer, policy_logvar, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    # no modification of the critic
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        intent_ph_n = make_intent_ph_n
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n + intent_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index], intent_ph_n[q_index]], 1)
        q,q_logvars = q_func(q_input, 1, policy_logvar ,scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + intent_ph_n+ [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n+ intent_ph_n, q)

        # # target network
        target_q, target_q_logvars = q_func(q_input, 1, policy_logvar,scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n + intent_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class TRPOMADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, act_traj_shape_n,intent_shape,  agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        self.kl = args.kl
        self.beta  1.0
        self.lr_multiplier = 1.0
        self.kl_targ = 0.003
        self.eta = 50
        self.policy_logvar = args.policy_logvar
        self.epochs = 20
        obs_ph_n = []
        act_traj_ph_n = []
        intent_ph_n = []
        make_old_target_ph = []
        make_log_vars_ph = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
            act_traj_ph_n.append(U.BatchInput(act_traj_shape_n[i], name = "action_trajectory"+str(i)).get())
            intent_ph_n.append(U.BatchInput(intent_shape[i], name = "intent"+str(i)).get())
            make_old_target_ph.append(U.BatchInput(intent_shape[i], name = 'old_target' + str(i)).get())
            make_log_vars_ph.append(U.BatchInput(intent_shape[i],name = 'log_vars'+str()).get())
        act_size = act_space_n[0].n
        make_beta_ph = tf.placeholder(tf.float32, (), 'beta')
        make_lr_ph = tf.placeholder(tf.float32, (), 'learning_rate')

        self.get_intent, self.i_train, self.i_trpo_object = i_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            intent_ph_n = intent_ph_n,
            act_space_n = act_space_n,
            make_act_traj_ph_n = act_traj_ph_n,
            make_intent_ph_n  =intent_ph_n,
            make_old_target_ph = make_old_target_ph,
            make_old_target_ph = make_old_target_ph,
            make_beta_ph = make_beta_ph,
            make_lr_ph = make_lr_ph,
            i_func = model,
            i_index = agent_index,
            output_size = (self.n-1) * act_size,
            policy_logvar = self.args.policy_logvar,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            reuse = False
            ) 
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            make_intent_ph_n = intent_ph_n,
            q_index=agent_index,
            q_func=model,
            policy_logvar = self.args.policy_logvar,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            make_intent_ph_n = intent_ph_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            policy_logvar = self.args.policy_logvar,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

    def intent(self, obs, act_traj):
        # print(np.array(act_traj).shape)
        # print(np.array(obs).shape)
        intent = self.get_intent(*( [[obs]] + [[act_traj]]) )[0]
        return intent

    def action(self, obs, intent):
        return self.act(*([[obs]] +[[intent]]))[0]

    def experience(self, obs, act, rew, new_obs, act_traj, intent, act_traj_next, intent_next, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, act_traj, intent,act_traj_next, intent_next,float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n_ep = []
        obs_next_n_ep = []
        act_n_ep = []
        act_traj_n_ep = []
        intent_n_ep = []
        act_traj_next_n_ep = []
        intent_next_n_ep = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next,act_traj, intent,act_traj_next, intent_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n_ep.append(obs)
            obs_next_n_ep.append(obs_next)
            act_n_ep.append(act)
            act_traj_n_ep.append(act_traj)
            intent_n_ep.append(intent)
            act_traj_next_n_ep.append(act_traj_next)
            intent_next_n_ep.append(intent_next)
        obs, act, rew, obs_next, act_traj, intent, act_traj_next, intent_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = len(obs_n_ep)
        target_q = 0.0
        
       

        for j in range(num_sample):
            target_act_next_n_j = [agents[i][j].p_debug['target_act'](*([obs_next_n[i][j]] +[intent_next_n[i][j]])) for i in range(self.n)]
            target_q_next_j = self.q_debug['target_q_values'](*(obs_next_n[j] + target_act_next_n_j[j] +intent_next_n[j]))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next_j
        target_q /= num_sample

        q_loss = self.q_train(*(obs_n + act_n +intent_n +[target_q]))

        self.q_update()

        #train i network offpolicy 
        # take the last actions from act_traj_next, use as the true label of the i 
        true_actions = get_true_action(act_traj_next_n)

        for e in range(self.epochs):
            # train p network and i network together, two source of gradient , 1 -- critic 2 -- supervision

            p_loss, p_kl, p_entropy = self.p_train(*(obs_n + act_n + intent_n + [self.lr_multiplier] + [self.beta]))

            i_loss, i_kl , i_entropy = self.i_train((*obs_n + act_traj_n + intent_n+ [self.lr_multiplier] + [self.beta]))
            if p_kl > self.kl_targ * 4 or i_kl > self.kl_targ:
                break

        if p_kl > self.kl_targ * 2 or i_kl > self.kl_targ * 4:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif p_kl < self.kl_targ / 2 or i_kl < self.kl_targ/4:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5        
    

        return [q_loss, p_loss,i_loss]
