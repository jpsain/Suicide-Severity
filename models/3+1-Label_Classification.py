#
# # Paper: "Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention"
#
# # 3+1-Label classification: Merge "Supportive" and "Indicator" classes
#

import csv
import string
from nltk import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from keras.utils.np_utils import to_categorical
import datetime, time
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, MaxPool2D
from keras.layers import Conv2D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers import Concatenate
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

punctuations = list(string.punctuation)

ip_txt_file = 'Data/500_Reddit_users_posts_labels.csv'  # CSV file: "User", "Post", "Label"
ip_feat_file = 'Data/External_Features.csv'             # CSV file: "User", "Features"

w2v_file = {'file': 'Data/english_conceptnet.txt', 'is_binary': False}

op_file = 'Result_3+1-Label_Classification.tsv'

severity_classes = {'Supportive': 0, 'Indicator': 1, 'Ideation': 2, 'Behavior': 3, 'Attempt': 4}

sys_params = {'emb_dim': 300,
              'max_sent_len': 1500,
              'str_padd': '@PADD',
              'cross_val': 5}

cnn_params = {'no_filters': 100,
              'kernels': [3, 4, 5],
              'channel': 1,
              'c_stride': (1, sys_params['emb_dim']),
              'pad': 'same',
              'ip_shape': (sys_params['max_sent_len'], sys_params['emb_dim'], 1),
              'c_activ': 'relu',
              'drop_rate': 0.3,
              'dense_1_unit': 128,
              'dense_2_unit': 128,
              'dense_activ': 'relu',
              'op_unit': 4,             # 4-Label classification (merging "Supportive" and "Indicator" class)
              'op_activ': 'softmax',
              'l_rate': 0.001,
              'loss': 'categorical_crossentropy',
              'batch': 4,
              'epoch': 50,
              'verbose': 1}

intermediate_layer = 'flat_drop'    # for extracting features from CNN

print '\nSystem Parameters: ', sys_params
print '\nCNN Parameters: ', cnn_params

# Read the input CSV file
def read_ip_file(ip_file, lst_merge_class=['Supportive', 'Indicator']):

    padd = sys_params['str_padd']
    max_len = sys_params['max_sent_len']

    x_data, y_data = [], []

    if ip_file:
        with open(ip_file) as csv_file:

            # Exclude the first line (header)
            csv_file.next()
            csv_reader = csv.reader(csv_file, delimiter=',')

            # Loop through each line
            for row in csv_reader:

                sent = row[1]

                # Remove non-ascii characters
                printable = set(string.printable)
                sent = filter(lambda x: x in printable, sent).lower()

                # Remove punctuation
                lst_tokens = [item.strip("".join(punctuations)) for item in word_tokenize(sent) if
                              item not in punctuations]

                # Strip the sentence if it exceeds the max length
                if len(lst_tokens) > max_len:
                    lst_tokens = lst_tokens[:max_len]

                # Padd the sentence if the length is less than max length
                elif len(lst_tokens) < max_len:
                    for j in range(len(lst_tokens), max_len):
                        lst_tokens.append(padd)

                label = row[2].strip()

                # if the label is 'Supportive'(label_no=0) or 'Indicator'(label_no=1), then add 0 as class_label
                # otherwise add (label_no - 1): eg. for label_no=2, add 1, the next class after 0;
                if label in lst_merge_class:
                    y_data.append(0)

                else:
                    y_data.append(severity_classes[row[2].strip()] - 1)

                x_data.append(lst_tokens)

    return x_data, y_data

# Vectorize the input data using pretrained word2vec embedding lookup
def vectorize_data(lst_input):

    padd = sys_params['str_padd']
    wv_size = sys_params['emb_dim']

    # Load the pre-trained word2vec model
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file['file'], binary=w2v_file['is_binary'])

    # Get the word2vec vocabulary
    vocab = w2v_model.vocab

    #
    padding_zeros = np.zeros(wv_size, dtype=np.float32)

    x_data = []

    # Loop through each sentence
    for sent in lst_input:
        emb = []
        for tok in sent:

            # Zero-padding for padded tokens
            if tok == padd:
                emb.append(list(padding_zeros))

            # Get the token embedding from the word2vec model
            elif tok in vocab:
                emb.append(w2v_model[tok].astype(float).tolist())

            # Zero-padding for out-of-vocab tokens
            else:
                emb.append(list(padding_zeros))

        x_data.append(emb)

    del w2v_model, vocab

    return np.array(x_data)

# Prepare the input data
def read_data(ip_file):

    # Read the input file
    x_data, y_data = read_ip_file(ip_file, lst_merge_class=['Supportive', 'Indicator'])

    # Vectorize the data
    x_data = vectorize_data(x_data)

    # # Reshape the data for CNN
    # x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)  # last argument 1 indicates #channel

    # Convert into numpy array
    x_data, y_data = np.array(x_data), np.array(y_data)

    return x_data, y_data

# Read additional external features
def read_external_features(input_file, feature_file):

    lst_users = []
    feature_dim = 16

    # Read the user_ids from the input file ["User", "Post", "Label"]
    with open(input_file) as f:
        for line in f:
            split = line.strip().split('\t')
            lst_users.append(split[0])

    features = []
    dct_user_featurs = {}

    with open(feature_file) as csv_file:
        # Exclude the header ['User', feature_scores ...]
        csv_file.next()

        # Read the CSV feature file
        csv_reader = csv.reader(csv_file, delimiter=',')

        # Loop through each row
        for row in csv_reader:
            # Start reading from 1-st value as 0-th value is "User"
            # convert the feature score into float
            scores = [float(val) for val in row[1:]]

            # Dictionary that maps: "User" --> "Feature"
            dct_user_featurs[row[0]] = scores

    # Read the features for user in the same sequence as in the input file
    # and add it to a list so that it can be merged easily the CNN generated features in the same sequence
    for user in lst_users:
        # If we have the features for the user then add it to the list
        if user in dct_user_featurs.keys():
            features.append(dct_user_featurs[user])

        # If we don't have features generated for a user, then add zeros
        else:
            features.append(list(np.zeros(feature_dim)))

    return np.array(features)

# Returns the CNN model
def get_cnn_model():
    seq_len = sys_params['max_sent_len']
    emb_dim = sys_params['emb_dim']

    l_ip = Input(shape=(seq_len, emb_dim, 1), dtype='float32')
    lst_convfeat = []
    for filter in cnn_params['kernels']:
        l_conv = Conv2D(filters=cnn_params['no_filters'], kernel_size=(filter, emb_dim), strides=cnn_params['c_stride'],
                        padding=cnn_params['pad'], data_format='channels_last', input_shape=cnn_params['ip_shape'],
                        activation=cnn_params['c_activ'])(l_ip)
        l_pool = MaxPool2D(pool_size=(seq_len, 1))(l_conv)
        lst_convfeat.append(l_pool)

    l_concat = Concatenate(axis=1)(lst_convfeat)
    l_flat = Flatten()(l_concat)
    l_drop = Dropout(rate=cnn_params['drop_rate'], name='flat_drop')(l_flat)

    # l_dense1 = Dense(units=cnn_params['dense_1_unit'], activation=cnn_params['dense_activ'], name='dense_1')(l_flat)
    # l_drop2 = Dropout(rate=cnn_params['drop_rate'])(l_dense1)

    # l_dense2 = Dense(units=cnn_params['dense_2_unit'], activation=cnn_params['dense_activ'], name='dense_2')(l_drop2)

    l_op = Dense(units=cnn_params['op_unit'], activation=cnn_params['op_activ'], name='cnn_op')(l_drop)

    final_model = Model(l_ip, l_op)
    final_model.compile(optimizer=Adam(lr=cnn_params['l_rate']), loss=cnn_params['loss'], metrics=['accuracy'])    # 'categorical_crossentropy'

    return final_model

# Returns a MLP model for final classification
def get_mlp_model(ip_dim):

    mlp_model = Sequential()

    mlp_model.add(Dense(units=cnn_params['op_unit'], activation=cnn_params['op_activ'], name='classif_op',
                            input_dim=ip_dim))
    mlp_model.compile(optimizer=Adam(lr=cnn_params['l_rate']), loss=cnn_params['loss'],
                          metrics=['accuracy'])
    return mlp_model

# Compute Precision, Recall, and F1-score
def get_prf1_score(y_true, y_pred):
    tp, fp, fn = 0.0, 0.0, 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            tp += 1
        elif y_pred[i] > y_true[i]:
            fp += 1
        else:
            fn += 1
    if tp == 0:
        tp = 1.0
    if fp == 0:
        fp = 1.0
    if fn == 0:
        fn  = 1.0
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    print '\nPrecision: {0}\t Recall: {1}\t F1-Score: {2}'.format(P, R, F)
    return {'P': P, 'R': R, 'F': F}


if __name__ == '__main__':

    with open(op_file, 'w') as of:

        x_data, y_data = read_data(ip_txt_file)

        ext_feature = read_external_features(ip_txt_file, ip_feat_file)

        cv_count = 0
        k_score = []

        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=sys_params['cross_val'])
        skf.get_n_splits(x_data, y_data)

        # Run the model for each splits
        for train_index, test_index in skf.split(x_data, y_data):
            cv_count += 1
            print '\nRunning Stratified Cross Validation: {0}/{1}...'.format(cv_count, sys_params['cross_val'])

            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            # Convert the class labels into categorical
            y_train, y_test = to_categorical(y_train), to_categorical(y_test)

            # Reshape the data for CNN
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

            # External features for this particular split
            train_ext_feat, test_ext_feat = ext_feature[train_index], ext_feature[test_index]

            # CNN model for training on the embedded text input
            cnn_model = get_cnn_model()
            print cnn_model.summary()

            # Train the model
            cnn_model.fit(x=x_train, y=y_train, batch_size=cnn_params['batch'], epochs=cnn_params['epoch'], verbose=cnn_params['verbose'])

            # Trained model for extracting features from intermediate layer
            model_feat_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(intermediate_layer).output)

            # Get CNN gerated features
            train_cnn_feat = model_feat_extractor.predict(x_train)
            test_cnn_feat = model_feat_extractor.predict(x_test)

            # Merge the CNN generated features with the external features
            x_train_features = []
            for index, cnn_feature in enumerate(train_cnn_feat):
                tmp_feat = list(cnn_feature)
                tmp_feat.extend(list(train_ext_feat[index]))
                x_train_features.append(np.array(tmp_feat))

            x_test_features = []
            for index, cnn_feature in enumerate(test_cnn_feat):
                tmp_feat = list(cnn_feature)
                tmp_feat.extend(list(test_ext_feat[index]))
                x_test_features.append(np.array(tmp_feat))

            # Convert the list into numpy array
            x_train_features = np.array(x_train_features)
            x_test_features = np.array(x_test_features)

            del train_cnn_feat, test_cnn_feat

            # Get the MLP model for final classification
            mlp_model = get_mlp_model(ip_dim = len(x_train_features[0]))
            print mlp_model.summary()

            tc = time.time()

            # Train the MLP model
            mlp_model.fit(x=x_train_features, y=y_train, batch_size=cnn_params['batch'], epochs=cnn_params['epoch'], verbose=cnn_params['verbose'])

            print '\nTime elapsed in training CNN: ', str(datetime.timedelta(seconds=time.time() - tc))
            del x_train, y_train

            print '\nEvaluating on Test data...\n'
            # # Print Loss and Accuracy
            model_metrics = mlp_model.evaluate(x_test_features, y_test)

            for i in range(len(model_metrics)):
                print mlp_model.metrics_names[i], ': ', model_metrics[i]

            y_pred = mlp_model.predict(x_test_features)

            y_pred = np.argmax(y_pred, axis=-1)
            y_test = np.argmax(y_test, axis=-1)

            # Scikit-learn classification report (P, R, F1, Support)
            report = classification_report(y_test, y_pred)
            print report

            of.write('Cross_Val:\n')
            for i in range(len(y_pred)):
                of.write('\t'.join([str(y_test[i]), str(y_pred[i])]) + '\n')

            score = get_prf1_score(y_test, y_pred)
            k_score.append(score)

        print k_score

        avgP = np.average([score['P'] for score in k_score])
        avgR = np.average([score['R'] for score in k_score])
        avgF = np.average([score['F'] for score in k_score])

        print '\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}'.format(avgP, avgR, avgF)
