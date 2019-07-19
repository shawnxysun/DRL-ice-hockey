import datetime
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
from nn.td_prediction_lstm_V3 import td_prediction_lstm_V3
from nn.td_prediction_lstm_V4 import td_prediction_lstm_V4
from utils import handle_trace_length, get_together_training_batch, compromise_state_trace_length
from configuration import MODEL_TYPE, MAX_TRACE_LENGTH, FEATURE_NUMBER, BATCH_SIZE, GAMMA, H_SIZE, \
    model_train_continue, FEATURE_TYPE, ITERATE_NUM, learning_rate, SPORT, directory_generated_Q_data, \
    save_mother_dir, action_all

ACTION_TO_MIMIC = 'assist'

SAVED_NETWORK = save_mother_dir + "/models/hybrid_sl_saved_NN/Scale-three-cut_together_saved_networks_feature" + str(
    FEATURE_TYPE) + "_batch" + str(
    BATCH_SIZE) + "_iterate" + str(
    ITERATE_NUM) + "_lr" + str(
    learning_rate) + "_" + str(MODEL_TYPE) + "_MaxTL" + str(MAX_TRACE_LENGTH)

Q_data_DIR = save_mother_dir + '/' + directory_generated_Q_data

DATA_STORE = "/Users/xiangyusun/Desktop/2019-icehockey-data-preprocessed/2018-2019"

DIR_GAMES_ALL = os.listdir(DATA_STORE)

timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
data_file_name = 'sportlogiq_data_' + ACTION_TO_MIMIC + '_' + str(timestamp) + '.txt'
data_description_file_name = 'sportlogiq_data_description_' + ACTION_TO_MIMIC + '_' + str(timestamp) + '.txt'

# read mean and variance for unstandarization
MEAN_FILE = DATA_STORE + '/feature_mean.txt'
VAR_FILE = DATA_STORE + '/feature_var.txt'

def write_Q_data_txt(fileWriter, Q_values, state_features, action_index):
    current_batch_size = len(Q_values)
    for batch_index in range(0, current_batch_size):
        Q_value = str(Q_values[batch_index][0]).strip() # only the Q_home for now

        # only consider HOME (index 9) for now
        if state_features[batch_index][0][9] > 0:

            # the first 12 elements are state features
            action_index_in_feature = 12 + action_index

            # generate the data only if the action of the current state is what we want
            if state_features[batch_index][0][action_index_in_feature] > 0:

                # flat the state features of all histories
                state_feature = ''
                # for history_index in range(0, len(state_features[batch_index])):
                for history_index in range(0, 1): # only consider the curent state
                    for feature_index in range(0, len(state_features[batch_index][history_index])):
                        # ignore actions of current state, since we only generate data for 1 action
                        if history_index == 0 and feature_index >= 12 and feature_index <= 44:
                            continue

                        # ignore home_away one hot of current state
                        if history_index == 0 and feature_index >= 9 and feature_index <= 10:
                            continue
                        
                        state_feature_value = state_features[batch_index][history_index][feature_index]

                        # # check if it is action
                        # if (feature_index-12-33) % 45 >= 12 and (feature_index-12-33) % 45 <= 44: 
                        #     if state_features[batch_index][history_index][feature_index] > 0:
                        #         state_feature_value = 1
                        #     else:
                        #         state_feature_value = 0

                        state_feature = state_feature + str(state_feature_value).strip() + ' '

                # TODO : state features are standarized before training, need to unstandarize them for mimic learning

                # write a line [Q, state_features_history_1, state_features_history_2, one_hot_action_history_2, ..., state_features_history_10, one_hot_action_history_10]
                fileWriter.write(Q_value.strip() + ' ' + state_feature.strip() + '\n')

def generate(sess, model, fileWriter, action_index):
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

        # skip the hidden mac file, feature_var.txt and feature_mean.txt
        if dir_game == '.DS_Store' or "feature_var" in dir_game or "feature_mean" in dir_game:
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

            write_Q_data_txt(fileWriter, Q_values, s_t0_batch, action_index)

            s_t0 = s_tl

            if terminal:
                break

def generation_start(fileWriter, action_index):
    sess = tf.InteractiveSession()
    if MODEL_TYPE == "v3":
        nn = td_prediction_lstm_V3(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    elif MODEL_TYPE == "v4":
        nn = td_prediction_lstm_V4(FEATURE_NUMBER, H_SIZE, MAX_TRACE_LENGTH, learning_rate)
    else:
        raise ValueError("MODEL_TYPE error")
    generate(sess, nn, fileWriter, action_index)

def generete_data_description_file():
    descriptionFileWriter = open(Q_data_DIR + '/' + data_description_file_name, 'w')
    # 3: data file name, NA, which line to start with
    # 1: Q
    # 12: the state features of current state, ignore actions
    # 45 * 9: (state features + one hot action) * 9 histories
    # 2: ingore home_away one hot
    # for line in range(0, 3 + 1 + 12 + 45 * 9):
    for line in range(0, 3 + 1 + 12 - 2):
        if line == 0:
            descriptionFileWriter.write(data_file_name + '\n')
        elif line == 1:
            descriptionFileWriter.write('NA\n')
        elif line ==2:
            descriptionFileWriter.write('1\n')
        elif line == 3:
            descriptionFileWriter.write('1 Q d\n')
        elif line == 4:
            descriptionFileWriter.write('2 xAdjCoord n\n')
        elif line == 5:
            descriptionFileWriter.write('3 yAdjCoord n\n')
        elif line == 6:
            descriptionFileWriter.write('4 scoreDifferential n\n')
        elif line == 7:
             descriptionFileWriter.write('5 manpowerSituation c\n')
        elif line == 8:
             descriptionFileWriter.write('6 outcome c\n')
        elif line == 9:
             descriptionFileWriter.write('7 velocity_x n\n')
        elif line == 10:
             descriptionFileWriter.write('8 velocity_y n\n')
        elif line == 11:
             descriptionFileWriter.write('9 time_remain n\n')
        elif line == 12:
             descriptionFileWriter.write('10 duration n\n')
        #  ignore home_away of current state
        # elif line == 13:
        #      descriptionFileWriter.write('11 home c\n')
        # elif line == 14:
        #      descriptionFileWriter.write('12 away c\n')
        elif line == 13:
             descriptionFileWriter.write('11 angle2gate n\n')

    descriptionFileWriter.close()

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

    generete_data_description_file()

    # the generated Q data file only contains data which has action 'ACTION_TO_MIMIC'
    action_index = action_all.index(ACTION_TO_MIMIC)

    fileWriter = open(Q_data_DIR + '/' + data_file_name, 'w')
    generation_start(fileWriter, action_index)
    fileWriter.close()
