import datetime
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
from nn.td_prediction_lstm_V3 import td_prediction_lstm_V3
from nn.td_prediction_lstm_V4 import td_prediction_lstm_V4
from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, directory_generated_Q_data, save_mother_dir

SAVED_NETWORK = save_mother_dir + "/models/hybrid_sl_saved_NN/Scale-three-cut_together_saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)

Q_data_DIR = save_mother_dir + '/' + directory_generated_Q_data

DATA_STORE = "/Users/xiangyusun/Desktop/2019-icehockey-data-preprocessed/2018-2019"

# read mean and variance for unstandarization
MEAN_FILE = DATA_STORE + '/feature_mean.txt'
VAR_FILE = DATA_STORE + '/feature_var.txt'

DIR_GAMES_ALL = os.listdir(DATA_STORE)

def write_Q_data_txt(fileWriter, Q_values, state_features):
    current_batch_size = len(Q_values)
    for batch_index in range(0, current_batch_size):
        Q_value = str(Q_values[batch_index][0]).strip() # only the Q_home for now

        # flat the state features of all histories
        state_feature = ''
        for history_index in range(0, len(state_features[batch_index])):
            for feature_index in range(0, len(state_features[batch_index][history_index])):
                state_feature = state_feature + str(state_features[batch_index][history_index][feature_index]).strip() + ' '

        # TODO : state features are standarized before training, need to unstandarize them for mimic learning

        # write a line [Q, state_features_history_1, one_hot_action_history_1, ..., state_features_history_10, one_hot_action_history_10]
        fileWriter.write(Q_value.strip() + ' ' + state_feature.strip() + '\n')

def generate(sess, model, fileWriter):
    # loading network
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state(SAVED_NETWORK)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Checkpoint loaded successfully:", checkpoint.model_checkpoint_path)
    else:
        raise Exception("Error: cannot load saved model.")
    
    # generate data with Q vaules using data from all subfolders in processed data folder
    for dir_game in DIR_GAMES_ALL:

        # skip the hidden mac file
        if dir_game == '.DS_Store':
            continue

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

            write_Q_data_txt(fileWriter, Q_values, s_t0_batch)

            s_t0 = s_tl

            if terminal:
                break

def generation_start(fileWriter):
    sess = tf.InteractiveSession()
    if MODEL_TYPE == "v3":
        nn = td_prediction_lstm_V3(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    elif MODEL_TYPE == "v4":
        nn = td_prediction_lstm_V4(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    else:
        raise ValueError("MODEL_TYPE error")
    generate(sess, nn, fileWriter)


if __name__ == '__main__':
    if not os.path.isdir(Q_data_DIR):
        os.mkdir(Q_data_DIR)

    # read mean and variance for unstandarization
    meanFileReader = open(MEAN_FILE, 'r')
    mean_str = meanFileReader.read().split(' ')
    meanFileReader.close()

    varFileReader = open(VAR_FILE, 'r')
    var_str = varFileReader.read().split(' ')
    varFileReader.close()

    means = []
    variances = []
    for i in range(0, FEATURE_NUMBER): # both mean and variance files have state_features + one_hot_action
        means.append(float(mean_str[i]))
        variances.append(float(var_str[i]))
    
    # TODO : need to modify the generated mean and variance files in preprocess.py, there are many zeros

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    fileWriter = open(Q_data_DIR + '/sportlogiq_data_' + str(timestamp) + '.txt', 'w')
    generation_start(fileWriter)
    fileWriter.close()
