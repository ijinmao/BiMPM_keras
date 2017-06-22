# -*- coding: utf-8 -*-
"""Model graph of Bilateral Multi-Perspective Matching.

References:
    Bilateral Multi-Perspective Matching for Natural Language Sentences
"""
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.layers.merge import concatenate
import keras.backend as K
from config import (
    BiMPMConfig, TrainConfig
)
from model.multi_perspective import MultiPerspective
from models.layers import (
    WordRepresLayer, CharRepresLayer, ContextLayer, PredictLayer
)

np.random.seed(BiMPMConfig.SEED)


def build_model(embedding_matrix, word_index, char_index=None):
    print('--- Building model...')

    # Parameters
    sequence_length = TrainConfig.MAX_SEQUENCE_LENGTH
    nb_per_word = TrainConfig.MAX_CHAR_PER_WORD
    rnn_unit = BiMPMConfig.RNN_UNIT
    nb_words = min(TrainConfig.MAX_NB_WORDS, len(word_index)) + 1
    word_embedding_dim = TrainConfig.WORD_EMBEDDING_DIM
    dropout = BiMPMConfig.DROP_RATE
    context_rnn_dim = BiMPMConfig.CONTEXT_LSTM_DIM
    mp_dim = BiMPMConfig.MP_DIM
    highway = BiMPMConfig.WITH_HIGHWAY
    aggregate_rnn_dim = BiMPMConfig.AGGREGATION_LSTM_DIM
    dense_dim = BiMPMConfig.DENSE_DIM
    if TrainConfig.USE_CHAR:
        nb_chars = min(TrainConfig.MAX_NB_CHARS, len(char_index)) + 1
        char_embedding_dim = TrainConfig.CHAR_EMBEDDING_DIM
        char_rnn_dim = TrainConfig.CHAR_LSTM_DIM

    # Model words input
    w1 = Input(shape=(sequence_length,), dtype='int32')
    w2 = Input(shape=(sequence_length,), dtype='int32')
    if TrainConfig.USE_CHAR:
        c1 = Input(shape=(sequence_length, nb_per_word), dtype='int32')
        c2 = Input(shape=(sequence_length, nb_per_word), dtype='int32')

    # Build word representation layer
    word_layer = WordRepresLayer(
        sequence_length, nb_words, word_embedding_dim, embedding_matrix)
    w_res1 = word_layer(w1)
    w_res2 = word_layer(w2)

    # Model chars input
    if TrainConfig.USE_CHAR:
        char_layer = CharRepresLayer(
            sequence_length, nb_chars, nb_per_word, char_embedding_dim,
            char_rnn_dim, rnn_unit=rnn_unit, dropout=dropout)
        c_res1 = char_layer(c1)
        c_res2 = char_layer(c2)
        sequence1 = concatenate([w_res1, c_res1])
        sequence2 = concatenate([w_res2, c_res2])
    else:
        sequence1 = w_res1
        sequence2 = w_res2

    # Build context representation layer
    context_layer = ContextLayer(
        context_rnn_dim, rnn_unit=rnn_unit, dropout=dropout, highway=highway,
        input_shape=(sequence_length, K.int_shape(sequence1)[-1],),
        return_sequences=True)
    context1 = context_layer(sequence1)
    context2 = context_layer(sequence2)

    # Build matching layer
    matching_layer = MultiPerspective(mp_dim)
    matching1 = matching_layer([context1, context2])
    matching2 = matching_layer([context2, context1])
    matching = concatenate([matching1, matching2])

    # Build aggregation layer
    aggregate_layer = ContextLayer(
        aggregate_rnn_dim, rnn_unit=rnn_unit, dropout=dropout, highway=highway,
        input_shape=(sequence_length, K.int_shape(matching)[-1],),
        return_sequences=False)
    aggregation = aggregate_layer(matching)

    # Build prediction layer
    pred = PredictLayer(dense_dim,
                        input_dim=K.int_shape(aggregation)[-1],
                        dropout=dropout)(aggregation)
    # Build model
    if TrainConfig.USE_CHAR:
        inputs = (w1, w2, c1, c2)
    else:
        inputs = (w1, w2)

    # Build model graph
    model = Model(inputs=inputs,
                  outputs=pred)

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model
