import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from models.bimpm import build_model as build_bimpm
from config import (
    DirConfig, TrainConfig, TestConfig, BiMPMConfig
)
from data_util import (
    get_text_sequence, save_training_history, create_submission,
    save_model, load_trained_models, load_word2vec_matrix,
    merge_several_folds_mean, split_train_data
)


def train_model():
    print('###### Start training for {}. ######'.format(
        'debugging' if DirConfig.DEBUG else 'production'))

    # Get model config
    config = BiMPMConfig

    # Load trained model from cache
    models = load_trained_models(config)
    if len(models) > 0:
        print('--- load model from cache.')
        # Compile model
        for m in models:
            m.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
        return models, None, None, None

    # Load train/test data set
    train_x1, train_x2, test_x1, test_x2, labels, test_ids, word_index, char_index = \
        get_text_sequence()

    # Load pretrained word embedding vectors
    embedding_matrix = load_word2vec_matrix(
        DirConfig.W2V_FILE, word_index, config)

    # Reweight params
    if TestConfig.RE_WEIGHT:
        class_weight = TestConfig.CLASS_WEIGHT
    else:
        class_weight = None

    # Split dataset indices
    kf = KFold(n_splits=10, shuffle=True)
    kf_gen = kf.split(labels)
    fold = 1
    models = []

    # Cross-validation train model
    for train_index, val_index in kf_gen:
        # Load current fold dataset
        train_data, train_labels, val_data, val_labels = split_train_data(
            train_x1, train_x2, labels, train_index, val_index)

        # Define validation sample weight
        val_weight = np.ones(len(val_labels))
        if TestConfig.RE_WEIGHT:
            val_weight *= TrainConfig.CLASS_WEIGHT[0]
            val_weight[val_labels == 0] = TrainConfig.CLASS_WEIGHT[1]

        # Build model
        model = build_model(embedding_matrix, word_index, char_index)

        # Define model callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(
            config.CHECKPOINT, save_best_only=True, save_weights_only=True)

        # Training
        history = model.fit(
            train_data, y=train_labels,
            validation_data=(val_data, val_labels, val_weight),
            # validation_split=TrainConfig.VALIDATION_SPLIT,
            epochs=TrainConfig.NB_EPOCH,
            batch_size=TrainConfig.BATCH_SIZE, shuffle=True,
            class_weight=class_weight,
            callbacks=[early_stopping, model_checkpoint])
        save_model(model, config, fold=fold)
        save_training_history(DirConfig.HISTORYA_DIR, config, history)
        fold += 1
        models.append(model)
        if fold > TrainConfig.KFOLD:
            break
    return models, test_x1, test_x2, test_ids


def test_model(model_name, models=[], test_x1=None, test_x2=None, test_ids=None):
    print('###### Start testing for {}. ######'.format(
        'debugging' if DirConfig.DEBUG else 'production'))

    config = BiMPMConfig

    # Load models from cache
    if len(models) == 0:
        models = load_trained_models(config)

    # Load test data from cache
    if test_x1 is None:
        _, _, test_x1, test_x2, _, test_ids, _, _ = \
            get_text_sequence()

    if TrainConfig.USE_CHAR:
        test_data = [test_x1[0], test_x2[0], test_x1[1], test_x2[1]]
    else:
        test_data = [test_x1, test_x2]

    predictions = []

    # Testing
    for model in models:
        preds = model.predict(
            test_data,
            batch_size=TestConfig.BATCH_SIZE, verbose=1)
        predictions.append(preds)

    preds_mean = np.array(merge_several_folds_mean(predictions, len(models)))
    create_submission(DirConfig.SUBM_DIR, config, preds_mean, test_ids)


def build_model(embedding_matrix, word_index, char_index):
    return build_bimpm(embedding_matrix, word_index, char_index)


def main():
    models, test_x1, test_x2, test_ids = train_model()
    test_model(models, test_x1, test_x2, test_ids)


if __name__ == '__main__':
    main()
