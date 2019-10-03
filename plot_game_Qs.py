import datetime
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from nn.td_prediction_lstm_V3 import td_prediction_lstm_V3
from nn.td_prediction_lstm_V4 import td_prediction_lstm_V4
from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, directory_generated_Q_data, \
    save_mother_dir, action_all

SAVED_NETWORK = save_mother_dir + "/models/hybrid_sl_saved_NN/Scale-three-cut_together_saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)

DATA_STORE = "/cs/oschulte/xiangyus/2019-icehockey-data-preprocessed/2018-2019"

DIR_GAMES_ALL = os.listdir(DATA_STORE)

def plot_game_Q_values(Q_values):
    event_numbers = range(0, len(Q_values))

    # Q_home = [Q_values[i]['home']for i in event_numbers]
    # Q_away = [Q_values[i]['away'] for i in event_numbers]
    # Q_end = [Q_values[i]['end'] for i in event_numbers]

    Q_home = [Q_values[i][0]for i in event_numbers]
    Q_away = [Q_values[i][1] for i in event_numbers]
    Q_end = [Q_values[i][2] for i in event_numbers]

    plt.figure()
    plt.plot(event_numbers, Q_home, label='home')
    plt.plot(event_numbers, Q_away, label='away')
    plt.plot(event_numbers, Q_end, label='end')
    plt.show()

def generate(sess, model, game_ID):
    # loading network
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Checkpoint loaded successfully:", checkpoint.model_checkpoint_path)
    else:
        raise Exception("Error: cannot load saved model.")
    
    dir_game = game_ID

    game_Q_values = None

    # find data file names
    game_files = os.listdir(DATA_STORE + "/" + dir_game)
    for filename in game_files:
        if "reward" in filename:
            reward_name = filename
        elif "state_feature_seq" in filename:
            state_input_name = filename
        elif "action_feature_seq" in filename:
                action_input_name = filename
        elif "lt" in filename:
            state_trace_length_name = filename

    reward = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + reward_name)
    reward = reward['reward'][0]

    state_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_input_name)
    state_input = (state_input['state_feature_seq'])

    action_input = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + action_input_name)
    action_input = (action_input['action_feature_seq'])

    state_trace_length = sio.loadmat(DATA_STORE + "/" + dir_game + "/" + state_trace_length_name)
    state_trace_length = (state_trace_length['lt'])[0]

    print("\n loaded files in folder " + str(dir_game) + " successfully")

    if len(state_input) != len(reward) or len(action_input) != len(reward) or len(state_trace_length) != len(reward):
            raise Exception('state/action length does not equal to reward length')

    train_len = len(state_input)
    train_number = 0
    # state representation is [state_features, one-hot-action]
    s_t0 = np.concatenate((state_input[train_number], action_input[train_number]), axis=1)
    train_number += 1

    while True:
        batch_return, train_number, s_tl = get_together_training_batch(s_t0,
        state_input,action_input,reward,train_number,train_len,state_trace_length,BATCH_SIZE)

        # get the batch variables
        s_t0_batch = [d[0] for d in batch_return]
        trace_t0_batch = [d[3] for d in batch_return]

        terminal = batch_return[len(batch_return) - 1][5]

        # calculate Q values
        Q_values = sess.run(model.read_out,
            feed_dict={
                model.trace_lengths: trace_t0_batch,
                model.rnn_input: s_t0_batch})

        if game_Q_values is None:
            game_Q_values = Q_values
        else:
            game_Q_values = np.concatenate((game_Q_values, Q_values))

        s_t0 = s_tl

        if terminal:
            break

    plot_game_Q_values(game_Q_values)

def start(game_ID):
    sess = tf.InteractiveSession()
    if MODEL_TYPE == "v3":
        nn = td_prediction_lstm_V3(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    elif MODEL_TYPE == "v4":
        nn = td_prediction_lstm_V4(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    else:
        raise ValueError("MODEL_TYPE error")
    generate(sess, nn, game_ID)

if __name__ == '__main__':    

    game_ID = '17080'

    start(game_ID)
