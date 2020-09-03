from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
import D2Denvironment_marl_test
import os
from replay_memory import ReplayMemory
import sys

my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth=True

class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = .99
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# ################## SETTINGS ######################        
length = 100
width = 100
vlb = 3
vup = 5
distancelb = 30
distanceup = 50
delta_theta = 5 # degree

numUsers = 4

IS_TRAIN = 0
IS_TEST = 1-IS_TRAIN

label = 'marl_model'
label_sarl = 'sarl_model'

env = D2Denvironment_marl_test.Environmnet(numUsers, width, length, vlb, vup, distancelb, distanceup, delta_theta)
env.new_game()

n_episode = 1000
n_step_per_episode = int(env.horizon/env.delta_tau)
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes


def get_state(env, idx=0, ind_episode=1., epsi= 0.02):
    
    load_remaining = env.data[idx] / env.data_size
    time_remaining = env.individual_time_limit[idx] / env.horizon
    # link_length = abs(env.linkDis[idx]-100)/100
    # link_stability_time = env.link_time[idx]-min(env.link_time[idx])
    # align_time = env.bAtime[idx]
    # link_stability_time = (env.link_time[idx]-min(env.link_time[idx]))/(max(env.link_time[idx])-min(env.link_time[idx]))
    # align_time = (env.bAtime[idx]-.015)/0.525
    # interference = (-env.intf[idx]-100)/100
    # interf_ind = (-env.intf_ind[idx,:]-30)/30
    # return np.concatenate(( interf_ind, interference, link_length, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate(( time_remaining, load_remaining, np.asarray([ind_episode, epsi])))

def get_state_sarl(env, idx=0, ind_episode=1., epsi= 0.02):
    
    load_remaining = env.data[idx] / env.data_size
    time_remaining = env.individual_time_limit[idx] / env.horizon
    # link_length = abs(env.linkDis[idx]-100)/100
    # link_stability_time = env.link_time[idx]-min(env.link_time[idx])
    # align_time = env.bAtime[idx]
    # link_stability_time = (env.link_time[idx]-min(env.link_time[idx]))/(max(env.link_time[idx])-min(env.link_time[idx]))
    # align_time = (env.bAtime[idx]-.015)/0.525
    # interference = (-env.intf[idx]-100)/100
    # interf_ind = (-env.intf_ind[idx,:]-30)/30
    # return np.concatenate(( interf_ind, interference, link_length, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate(( time_remaining, load_remaining, np.asarray([ind_episode, epsi])))




n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
n_input = len(get_state(env=env))
n_output = len(D2Denvironment_marl_test.Antenna().beamwidth)

g = tf.Graph()
with g.as_default():
    # ============== Training network ========================
    x = tf.placeholder(tf.float32, [None, n_input])

    w_1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4 = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4 = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w_1), b_1))
    layer_1_b = tf.layers.batch_normalization(layer_1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = tf.layers.batch_normalization(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = tf.layers.batch_normalization(layer_3)
    y = tf.nn.relu(tf.add(tf.matmul(layer_3_b, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)

    # compute loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')
    optim = tf.train.RMSPropOptimizer(learning_rate=1E-4, momentum=0.95, epsilon=0.01).minimize(g_loss)

    # ==================== Prediction network ========================
    x_p = tf.placeholder(tf.float32, [None, n_input])

    w_1_p = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2_p = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3_p = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4_p = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1_p = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2_p = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3_p = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4_p = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1_p = tf.nn.relu(tf.add(tf.matmul(x_p, w_1_p), b_1_p))
    layer_1_p_b = tf.layers.batch_normalization(layer_1_p)

    layer_2_p = tf.nn.relu(tf.add(tf.matmul(layer_1_p_b, w_2_p), b_2_p))
    layer_2_p_b = tf.layers.batch_normalization(layer_2_p)

    layer_3_p = tf.nn.relu(tf.add(tf.matmul(layer_2_p_b, w_3_p), b_3_p))
    layer_3_p_b = tf.layers.batch_normalization(layer_3_p)

    y_p = tf.nn.relu(tf.add(tf.matmul(layer_3_p_b, w_4_p), b_4_p))

    g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
    target_q_with_idx = tf.gather_nd(y_p, g_target_q_idx)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    
def predict(sess, s_t, ep, test_ep = False):
    action_number = len(env.action_list)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(action_number)
    else:
        pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action

def predict_sarl(sess, s_t):
    pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action


def q_learning_mini_batch(current_agent, current_sess):
    """ Training a sampled mini-batch """

    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()

    if current_agent.double_q:  # double q-learning
        pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
        q_t_plus_1 = current_sess.run(target_q_with_idx, {x_p: batch_s_t_plus_1, g_target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
        batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
    else:
        q_t_plus_1 = current_sess.run(y_p, {x_p: batch_s_t_plus_1})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    _, loss_val = current_sess.run([optim, g_loss], {g_target_q_t: batch_target_q_t, g_action: batch_action, x: batch_s_t})
    return loss_val

def update_target_q_network(sess):
    """ Update target q network once in a while """

    sess.run(w_1_p.assign(sess.run(w_1)))
    sess.run(w_2_p.assign(sess.run(w_2)))
    sess.run(w_3_p.assign(sess.run(w_3)))
    sess.run(w_4_p.assign(sess.run(w_4)))

    sess.run(b_1_p.assign(sess.run(b_1)))
    sess.run(b_2_p.assign(sess.run(b_2)))
    sess.run(b_3_p.assign(sess.run(b_3)))
    sess.run(b_4_p.assign(sess.run(b_4)))


def save_models(sess, model_path):
    """ Save models to the current directory with the name filename """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)


def load_models(sess, model_path):
    """ Restore models from the current directory with the name filename """

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    saver.restore(sess, model_path)


def print_weight(sess, target=False):
    """ debug """

    if not target:
        print(sess.run(w_1[0, 0:4]))
    else:
        print(sess.run(w_1_p[0, 0:4]))
        
# --------------------------------------------------------------
agents = []
sesses = []
for ind_agent in range(numUsers):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

    sess = tf.Session(graph=g,config=my_config)
    sess.run(init)
    sesses.append(sess)

agent_sarl = Agent(memory_entry_size=len(get_state(env)))
sess_sarl = tf.Session(graph=g,config=my_config)
sess_sarl.run(init)




# --------------------Testing ---------------------------------

if IS_TEST:
    print("\nRestoring the model...")

    for i in range(numUsers):
        model_path = label + '/agent_' + str(i)
        load_models(sesses[i], model_path)
    
    # restore the single-agent model
    model_path_single = label_sarl + '/agent'
    load_models(sess_sarl, model_path_single)
    
    D2D_success_list = []
    D2D_success_list_rand = []
    
    D2D_success_list_sarl = []
    
    rate_marl = np.zeros([n_episode_test, n_step_per_episode, numUsers])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, numUsers])
    data_marl = env.data_size * np.ones([n_episode_test, n_step_per_episode+1, numUsers])
    data_rand = env.data_size * np.ones([n_episode_test, n_step_per_episode+1, numUsers])
    
    action_all_testing_sarl = np.zeros(numUsers, dtype='int32')
    
    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_trajectory()
        # env.renew_channel()

        env.data = env.data_size * np.ones((numUsers,1))
        env.individual_time_limit = env.horizon * np.ones((numUsers,1))
        env.active_links = np.ones((numUsers,1), dtype='bool')

        env.data_rand = env.data_size * np.ones((numUsers,1))
        env.individual_time_limit_rand = env.horizon * np.ones((numUsers,1))
        env.active_links_rand = np.ones((numUsers,1), dtype='bool')
        
        env.data_rand_sarl = env.data_size * np.ones((numUsers,1))
        env.individual_time_limit_rand_sarl = env.horizon * np.ones((numUsers,1))
        env.active_links_rand_sarl = np.ones((numUsers,1), dtype='bool')
        
        for test_step in range(n_step_per_episode):
            # trained models
            action_all_testing = np.zeros(numUsers, dtype='int32')
            for i in range(numUsers):
                    state_old = get_state(env, i, 1, epsi_final)
                    action = predict(sesses[i], state_old, epsi_final, True)
                    action_all_testing[i] = action 

            action_temp = action_all_testing.copy()
            D2Drate, D2D_success = env.act_for_testing(action_temp)
            rate_marl[idx_episode, test_step,:] = np.ravel(D2Drate)
            data_marl[idx_episode, test_step+1,:] = np.ravel(env.data)

            # random baseline
            # action_rand = np.zeros(numUsers, dtype='int32')
            # action_rand = np.random.randint(0, len(env.action_list), numUsers) # band
            action_rand = 3*np.ones(numUsers, dtype='int32')

            D2D_rate_rand, D2D_success_rand = env.act_for_testing_rand(action_rand)
            rate_rand[idx_episode, test_step, :] = np.ravel(D2D_rate_rand)
            data_rand[idx_episode, test_step+1,:] = np.ravel(env.data_rand)
            
            # SARL
            i = test_step % numUsers
            state_sarl = get_state_sarl(env, i, 1, epsi_final)
            action = predict_sarl(sess_sarl, state_sarl)
            action_all_testing_sarl[i] = action
            
            action_temp_sarl = action_all_testing_sarl.copy()
            D2Drate_sarl, D2D_success_sarl = env.act_for_testing_sarl(action_temp_sarl)
            
            
            # update the environment and compute interference
            env.renew_trajectory()
            
            if test_step == n_step_per_episode - 1:
                D2D_success_list.append(D2D_success)
                D2D_success_list_rand.append(D2D_success_rand)
                D2D_success_list_sarl.append(D2D_success_sarl)
                
            
        print('marl', D2D_success_list[idx_episode], 'sarl', D2D_success_list_sarl[idx_episode], 'rand', D2D_success_list_rand[idx_episode])
            
            
    print('-------- marl -------------')
    print('numUsers:', numUsers)
    print('Pr(D2D success):', round(np.average(D2D_success_list), 4))
    #
    print('-------- sarl -------------')
    print('numUsers:', numUsers)
    print('Pr(D2D success):', round(np.average(D2D_success_list_sarl), 4))

    print('-------- random -------------')
    print('numUsers:', numUsers)
    print('Pr(D2D success):', round(np.average(D2D_success_list_rand), 4))


# The name "DPRA" is used for historical reasons. Not really the case...

    with open("Data.txt", "a") as f:
        f.write('-------- marl, ' + label + '------\n')
        f.write('numUsers: ' + str(numUsers) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(D2D_success_list), 5)) + '\n')
        f.write('-------- sarl, ' + label_sarl + '------\n')
        f.write('numUsers: ' + str(numUsers) + '\n')
        f.write('Pr(V2V): ' + str(round(np.average(D2D_success_list_sarl), 5)) + '\n')
        f.write('--------random ------------\n')
        f.write('Rand Pr(V2V): ' + str(round(np.average(D2D_success_list_rand), 5)) + '\n')


    current_dir = os.path.dirname(os.path.realpath(__file__))
    marl_path = os.path.join(current_dir, "model/" + label + '/rate_marl.mat')
    scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
    rand_path = os.path.join(current_dir, "model/" + label + '/rate_rand.mat')
    scipy.io.savemat(rand_path, {'rate_rand': rate_rand})

    data_marl_path = os.path.join(current_dir, "model/" + label + '/data_marl.mat')
    scipy.io.savemat(data_marl_path, {'data_marl': data_marl})
    demand_rand_path = os.path.join(current_dir, "model/" + label + '/data_rand.mat')
    scipy.io.savemat(demand_rand_path, {'data_rand': data_rand})


# close sessions
for sess in sesses:
    sess.close()
                
                