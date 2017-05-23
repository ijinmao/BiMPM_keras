import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from config import (
    DirConfig, TrainConfig
)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
import datetime
import os


def get_text_sequence():
    if os.path.isfile(DirConfig.CHAR1_CACHE_TRAIN):
        print('---- Load data from cache.')
        train_x1 = np.load(open(DirConfig.Q1_CACHE_TRAIN, 'rb'))
        train_x2 = np.load(open(DirConfig.Q2_CACHE_TRAIN, 'rb'))
        test_x1 = np.load(open(DirConfig.Q1_CACHE_TEST, 'rb'))
        test_x2 = np.load(open(DirConfig.Q2_CACHE_TEST, 'rb'))
        labels = np.load(open(DirConfig.TARGETS_CACHE, 'rb'))
        test_ids = np.load(open(DirConfig.TEST_ID_CACHE, 'rb'))
        word_index = np.load(open(DirConfig.WORD_INDEX_CACHE, 'rb')).item()
        char_index = None

        # use char representation
        if TrainConfig.USE_CHAR:
            train_words1 = np.load(open(DirConfig.CHAR1_CACHE_TRAIN, 'rb'))
            train_words2 = np.load(open(DirConfig.CHAR2_CACHE_TRAIN, 'rb'))
            test_words1 = np.load(open(DirConfig.CHAR1_CACHE_TEST, 'rb'))
            test_words2 = np.load(open(DirConfig.CHAR2_CACHE_TEST, 'rb'))
            char_index = np.load(open(DirConfig.CHAR_INDEX_CACHE, 'rb')).item()
    else:
        # load data from csv
        if DirConfig.DEBUG:
            train_data = pd.read_csv(DirConfig.SAMPLE_TRAIN_FILE)
            test_data = pd.read_csv(DirConfig.SAMPLE_TEST_FILE)
        else:
            train_data = pd.read_csv(DirConfig.TRAIN_FILE)
            test_data = pd.read_csv(DirConfig.TEST_FILE)

        # train and text text
        train_ori1 = list(train_data.question1.values.astype(str))
        train_ori2 = list(train_data.question2.values.astype(str))
        test_ori1 = list(test_data.question1.values.astype(str))
        test_ori2 = list(test_data.question2.values.astype(str))

        # target labels
        labels = train_data.is_duplicate.values
        test_ids = test_data.test_id
        np.save(open(DirConfig.TARGETS_CACHE, 'wb'), labels)
        np.save(open(DirConfig.TEST_ID_CACHE, 'wb'), test_ids)

        train_ori1 = preprocess_texts(train_ori1)
        train_ori2 = preprocess_texts(train_ori2)
        test_ori1 = preprocess_texts(test_ori1)
        test_ori2 = preprocess_texts(test_ori2)

        train_x1, train_x2, test_x1, test_x2, word_index = \
            get_word_seq(train_ori1, train_ori2, test_ori1, test_ori2)

        if TrainConfig.USE_CHAR:
            train_words1, train_words2, test_words1, test_words2, char_index = \
                get_char_seq(train_ori1, train_ori2, test_ori1, test_ori2)
        else:
            char_index = None

    if TrainConfig.USE_CHAR:
        # concatenate inputs
        train_x1 = (train_x1, train_words1)
        train_x2 = (train_x2, train_words2)
        test_x1 = (test_x1, test_words1)
        test_x2 = (test_x2, test_words2)

    return train_x1, train_x2, test_x1, test_x2, labels, test_ids, word_index, char_index


def get_word_seq(train_ori1, train_ori2, test_ori1, test_ori2):
    # fit tokenizer
    tk = Tokenizer(num_words=TrainConfig.MAX_NB_WORDS)
    tk.fit_on_texts(train_ori1 + train_ori2 + test_ori1 + test_ori2)
    word_index = tk.word_index

    # q1, q2 training text sequence
    # (sentence_len, MAX_SEQUENCE_LENGTH)
    train_x1 = tk.texts_to_sequences(train_ori1)
    train_x1 = pad_sequences(train_x1, maxlen=TrainConfig.MAX_SEQUENCE_LENGTH)
    train_x2 = tk.texts_to_sequences(train_ori2)
    train_x2 = pad_sequences(train_x2, maxlen=TrainConfig.MAX_SEQUENCE_LENGTH)

    # q1, q2 testing text sequence
    test_x1 = tk.texts_to_sequences(test_ori1)
    test_x1 = pad_sequences(test_x1, maxlen=TrainConfig.MAX_SEQUENCE_LENGTH)
    test_x2 = tk.texts_to_sequences(test_ori2)
    test_x2 = pad_sequences(test_x2, maxlen=TrainConfig.MAX_SEQUENCE_LENGTH)

    np.save(open(DirConfig.Q1_CACHE_TRAIN, 'wb'), train_x1)
    np.save(open(DirConfig.Q2_CACHE_TRAIN, 'wb'), train_x2)
    np.save(open(DirConfig.Q1_CACHE_TEST, 'wb'), test_x1)
    np.save(open(DirConfig.Q2_CACHE_TEST, 'wb'), test_x2)
    np.save(open(DirConfig.WORD_INDEX_CACHE, 'wb'), word_index)
    return train_x1, train_x2, test_x1, test_x2, word_index


def words_to_char_sequence(words_list, tk):
    """Convert words list to chars sequence

    # Arguments
        words: word list, (sentence_len, word_len)

    # Output shape
        (sentence_len, MAX_SEQUENCE_LENGTH, MAX_CHAR_PER_WORD)
    """
    c_seqs = np.zeros((len(words_list),
                       TrainConfig.MAX_SEQUENCE_LENGTH,
                       TrainConfig.MAX_CHAR_PER_WORD), dtype='int32')
    for w_i in xrange(len(words_list)):
        words = words_list[w_i]
        fixed_ws = np.zeros((TrainConfig.MAX_SEQUENCE_LENGTH,
                             TrainConfig.MAX_CHAR_PER_WORD), dtype='int32')
        ws = tk.texts_to_sequences(words)
        ws = pad_sequences(ws, maxlen=TrainConfig.MAX_CHAR_PER_WORD)
        if TrainConfig.MAX_SEQUENCE_LENGTH < len(words):
            max_word_len = TrainConfig.MAX_SEQUENCE_LENGTH
        else:
            max_word_len = len(words)
        fixed_ws[:max_word_len, :] = ws[:max_word_len, :]
        c_seqs[w_i] = fixed_ws
    return c_seqs


def get_char_seq(train_ori1, train_ori2, test_ori1, test_ori2):
    # extract words from each text
    train_words1 = extract_words(train_ori1)
    train_words2 = extract_words(train_ori2)
    test_words1 = extract_words(test_ori1)
    test_words2 = extract_words(test_ori2)

    # fit tokenizer
    tk = Tokenizer(num_words=TrainConfig.MAX_NB_CHARS, char_level=True)
    tk.fit_on_texts(train_ori1 + train_ori2 + test_ori1 + test_ori2)
    char_index = tk.word_index

    # q1, q2 training word sequence
    train_s1 = words_to_char_sequence(train_words1, tk)
    train_s2 = words_to_char_sequence(train_words2, tk)

    # q1, q2 testing word sequence
    test_s1 = words_to_char_sequence(test_words1, tk)
    test_s2 = words_to_char_sequence(test_words2, tk)

    # save cache
    np.save(open(DirConfig.CHAR1_CACHE_TRAIN, 'wb'), train_s1)
    np.save(open(DirConfig.CHAR2_CACHE_TRAIN, 'wb'), train_s2)
    np.save(open(DirConfig.CHAR1_CACHE_TEST, 'wb'), test_s1)
    np.save(open(DirConfig.CHAR2_CACHE_TEST, 'wb'), test_s2)
    np.save(open(DirConfig.CHAR_INDEX_CACHE, 'wb'), char_index)
    return train_s1, train_s2, test_s1, test_s2, char_index


# from https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):    
    # Convert words to lower case and split them
    text = str(text).lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)

    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)


def preprocess_texts(texts):
    processed = []
    for t in texts:
        processed.append(text_to_wordlist(
            t, remove_stopwords=TrainConfig.REMOVE_STOPWORDS, stem_words=TrainConfig.USE_STEM))
    return processed


def split_train_data(train_x1, train_x2, labels, train_index, val_index):
    if TrainConfig.USE_CHAR:
        train_w1 = train_x1[0][train_index]
        train_w2 = train_x2[0][train_index]
        train_c1 = train_x1[1][train_index]
        train_c2 = train_x2[1][train_index]
        train_data = [train_w1, train_w2, train_c1, train_c2]

        val_w1 = train_x1[0][val_index]
        val_w2 = train_x2[0][val_index]
        val_c1 = train_x1[1][val_index]
        val_c2 = train_x2[1][val_index]
        val_data = [val_w1, val_w2, val_c1, val_c2]
    else:
        train_data = [train_x1[train_index], train_x2[train_index]]
        val_data = [train_x1[val_index], train_x2[val_index]]

    train_labels = labels[train_index]
    val_labels = labels[val_index]
    return train_data, train_labels, val_data, val_labels


def extract_words(sentences):
    """Extract chars from each sentence

    # Arguments
        sentences: list of sentences
    """
    w_seqs = []
    for s in sentences:
        s = re.sub(r"[?^,!.\/'+-=()]", " ", s)
        s = s.strip()
        words = []
        for word in re.split('\\s+', s):
            words.append(word)
        w_seqs.append(words)
    return w_seqs


def load_word_embedding(type, vec_file, word_index, config):
    if type == 'glove':
        return load_glove_matrix(vec_file, word_index, config)
    else:
        return load_word2vec_matrix(vec_file, word_index, config)


def load_glove_matrix(vec_file, word_index, config):
    if os.path.isfile(DirConfig.GLOVE_CACHE):
        print('---- Load word vectors from cache.')
        embedding_matrix = np.load(open(DirConfig.GLOVE_CACHE, 'rb'))
        return embedding_matrix

    print('---- loading glove ...')
    embeddings_index = {}
    f = open(vec_file)
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    nb_words = min(config.MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, config.WORD_EMBEDDING_DIM))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # check the words which not in embedding vectors
    not_found_words = []
    for word, i in word_index.items():
        if word not in embeddings_index:
            not_found_words.append(word)

    np.save(open(DirConfig.GLOVE_CACHE, 'wb'), embedding_matrix)
    return embedding_matrix


def load_word2vec_matrix(vec_file, word_index, config):
    if os.path.isfile(DirConfig.W2V_CACHE):
        print('---- Load word vectors from cache.')
        embedding_matrix = np.load(open(DirConfig.W2V_CACHE, 'rb'))
        return embedding_matrix

    print('---- loading word2vec ...')
    word2vec = KeyedVectors.load_word2vec_format(
        vec_file, binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    nb_words = min(config.MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, config.WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % \
          np.sum(np.sum(embedding_matrix, axis=1) == 0))

    # check the words which not in embedding vectors
    not_found_words = []
    for word, i in word_index.items():
        if word not in word2vec.vocab:
            not_found_words.append(word)

    np.save(open(DirConfig.W2V_CACHE, 'wb'), embedding_matrix)
    return embedding_matrix


def save_training_history(path, config, history, fold=0):
    values = np.array(history.history.values())
    results = pd.DataFrame(values.transpose(), columns=[history.history.keys()])
    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    path = os.path.join(
        path, 'his_{}_trial_{}_db_{}_k_{}-{}.csv'.format(
            config.INFO, config.TRIAL, DirConfig.DEBUG, fold, suffix))
    results.to_csv(path)


def create_submission(path, config, preds, test_ids, low_threhold=0.05):
    print('----- Create submission for {}'.format(config.MODEL))
    if preds.shape[1] > 1:
        preds = preds[:, 1]
    preds = preds.clip(low_threhold, 1.0 - low_threhold)
    submission = pd.DataFrame(test_ids, columns=['test_id'])
    submission.loc[:, 'is_duplicate'] = preds.ravel()
    now = datetime.datetime.now()
    subm_file = os.path.join(path, 'subm_{}_trial_{}_db_{}-{}.csv'.format(
        config.INFO, config.TRIAL, DirConfig.DEBUG, str(now.strftime("%Y-%m-%d-%H-%M"))))
    submission.to_csv(subm_file, index=False)
    return subm_file


def save_model(model, config, fold=0):
    m_file = os.path.join(
        config.BASE_DIR, '{}_trial_{}_db_{}_k_{}_model.h5'.format(
            config.INFO, config.TRIAL, DirConfig.DEBUG, fold))
    w_file = os.path.join(
        config.BASE_DIR, '{}_trial_{}_db_{}_k_{}_weight.h5'.format(
            config.INFO, config.TRIAL, DirConfig.DEBUG, fold))
    model.save(m_file)
    model.save_weights(w_file)
    print('--- Saved model.')


def load_keras_model(config, custom_objects=None, fold=0):
    m_file = os.path.join(
        config.BASE_DIR, '{}_trial_{}_db_{}_k_{}_model.h5'.format(
            config.INFO, config.TRIAL, DirConfig.DEBUG, fold))
    if os.path.isfile(m_file):
        model = load_model(m_file, custom_objects)
        return model
    else:
        return None


def merge_several_folds_mean(data, nfolds):
    print('------ Merge several folds results to mean. -----')
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def load_trained_models(config):
    models = []
    for k in range(TrainConfig.KFOLD):
        model = load_keras_model(config, fold=k + 1)
        if model is None:
            break
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
        models.append(model)
    return models
