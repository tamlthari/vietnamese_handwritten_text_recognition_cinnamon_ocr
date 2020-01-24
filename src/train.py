#coding=utf-8
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

import h5py
import numpy as np
import unicodedata
import json
import os
import pathlib
import cv2
import datetime
import argparse
import itertools

import preprocess as prep
import model as mb
import evaluation

charset_base = "¶ #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstwuvxyzÂÊÔàáâãèéêẹìíòóôõùúýăĐđĩũƠơưạảấầẩẫậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
vocab_size = len(charset_base)
MAX_LABEL_LENGTH = 128
INPUT_SIZE = (2048, 128, 1)
PAD_TK = "¶"


BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


def text_to_labels(text):
    return np.asarray(list(map(lambda x: charset_base.index(x), text)), dtype=np.uint8)

def labels_to_text(labels):
    return ''.join(list(map(lambda x: charset_base[x] if x < len(charset_base) else "", labels)))

def get_label(type_):
    labels = json.load(open(os.path.join('..', 'data', '{}.json'.format(type_))))
    return labels

def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)/255
    image = tf.image.per_image_standardization(image)
    return image
    

def load_and_preprocess_image(path, label):
    # label = text_to_labels(label)
    # label = pad_sequences(label, maxlen=MAX_LABEL_LENGTH, padding='post')
    return preprocess_image(path), label

def cv2_augmentation(img,
                    rotation_range=3,
                    scale_range=0,
                    height_shift_range=5,
                    width_shift_range=5,
                    dilate_range=3,
                    erode_range=3):
    """Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)"""

    img = img.numpy().astype(np.float32)
    h, w = img.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range + 10)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    img = cv2.warpAffine(img, affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
    img = cv2.erode(img, erode_kernel, iterations=1)
    img = cv2.dilate(img, dilate_kernel, iterations=1)

    return img

@tf.function
def augmentation(image, label):
    image = tf.py_function(cv2_augmentation, [image], tf.float32)
    return image, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=100, augment=False):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)


    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)

# Switching repeat and batch because of tensor shape error
    # Repeat forever
    # ds = ds.batch(BATCH_SIZE)
    # ds = ds.repeat()

    if augment:
        ds.map(augmentation, num_parallel_calls=AUTOTUNE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

def prepare_for_testing(ds):
    # Switching repeat and batch because of tensor shape error
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

def build_dataset(type_, cache=False, augment=False, training=True):
    DATA_FOLDER = os.path.join('..', 'data', type_)
    ds = tf.data.Dataset.list_files(os.path.join(DATA_FOLDER, '*'))
    labels = get_label(type_)
    
    all_image_paths = [str(item) for item in pathlib.Path(DATA_FOLDER).glob('*') if item.name in labels]
    labels = [labels[pathlib.Path(path).name] for path in all_image_paths]
    labels = [prep.text_standardize(label) for label in labels]
    all_image_labels = [text_to_labels(label) for label in labels]
    all_image_labels = pad_sequences(all_image_labels, maxlen=MAX_LABEL_LENGTH, padding='post')
    n_samples = len(all_image_labels)
    steps_per_epoch = tf.math.ceil(n_samples/BATCH_SIZE)

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    if training:
        ds = prepare_for_training(ds, cache=cache, shuffle_buffer_size=n_samples, augment=True)
    else:
        ds = prepare_for_testing(ds)

    return ds, steps_per_epoch, labels

def decode_batch(prediction):
    result = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        result.append(outstr)
    return result
#     print(prediction.shape)
#     return K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
#                          greedy=True)[0][0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainattn", action="store_true", default=False)
    parser.add_argument("--trainquoc", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--path", type=str, required=False) # path to test data
    args = parser.parse_args()
    checkpoint = './checkpoint_weights.hdf5'

    #Training with Quoc's model
    if args.trainquoc:
        train_ds, num_steps_train, _ = build_dataset('train', cache=True)
        test_ds, num_steps_val, _ = build_dataset('test', training=False)
        model = mb.build_model_quoc(input_size=INPUT_SIZE, d_model=vocab_size+1, learning_rate=0.001)
#         model.load_weights(checkpoint)
        model.summary()
        batch_stats_callback = mb.CollectBatchStats()
        callbacks = mb.callbacks
        start_time = datetime.datetime.now()

        h = model.fit(train_ds,
                    steps_per_epoch = num_steps_train,
                    epochs=100,
                    validation_data = test_ds,
                    validation_steps = num_steps_val,
                    callbacks=callbacks)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))

        t_corpus = "\n".join([
            "Batch:                   {}\n".format(BATCH_SIZE),
            "Time per epoch:          {}".format(time_epoch),
            "Total epochs:            {}".format(len(loss)),
            "Best epoch               {}\n".format(min_val_loss_i + 1),
            "Training loss:           {}".format(loss[min_val_loss_i]),
            "Validation loss:         {}".format(min_val_loss),
        ])

        with open(os.path.join("train_stats.txt"), "w") as f:
            f.write(t_corpus)
            print(t_corpus)

    #Training with attention
    if args.trainattn:
        train_ds, num_steps_train, _ = build_dataset('train', cache=True)
        test_ds, num_steps_val, _ = build_dataset('test', training=False)
        model = mb.build_model(input_size=INPUT_SIZE, d_model=vocab_size+1, learning_rate=0.001)
#         model.load_weights(checkpoint)
        model.summary()
        batch_stats_callback = mb.CollectBatchStats()
        callbacks = mb.callbacks
        start_time = datetime.datetime.now()

        h = model.fit(train_ds,
                    steps_per_epoch = num_steps_train,
                    epochs=100,
                    validation_data = test_ds,
                    validation_steps = num_steps_val,
                    callbacks=callbacks)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))

        t_corpus = "\n".join([
            "Batch:                   {}\n".format(BATCH_SIZE),
            "Time per epoch:          {}".format(time_epoch),
            "Total epochs:            {}".format(len(loss)),
            "Best epoch               {}\n".format(min_val_loss_i + 1),
            "Training loss:           {}".format(loss[min_val_loss_i]),
            "Validation loss:         {}".format(min_val_loss),
        ])

        with open(os.path.join("train_stats.txt"), "w") as f:
            f.write(t_corpus)
            print(t_corpus)

    if args.train:
        train_ds, num_steps_train, _ = build_dataset('train', cache=True)
        test_ds, num_steps_val, _ = build_dataset('test', training=False)
        model = mb.build_model_woattn(input_size=INPUT_SIZE, d_model=vocab_size+1, learning_rate=0.001)
#         model.load_weights(checkpoint)
        model.summary()
        batch_stats_callback = mb.CollectBatchStats()
        callbacks = mb.callbacks
        start_time = datetime.datetime.now()

        h = model.fit(train_ds,
                    steps_per_epoch = num_steps_train,
                    epochs=100,
                    validation_data = test_ds,
                    validation_steps = num_steps_val,
                    callbacks=callbacks)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))

        t_corpus = "\n".join([
            "Batch:                   {}\n".format(BATCH_SIZE),
            "Time per epoch:          {}".format(time_epoch),
            "Total epochs:            {}".format(len(loss)),
            "Best epoch               {}\n".format(min_val_loss_i + 1),
            "Training loss:           {}".format(loss[min_val_loss_i]),
            "Validation loss:         {}".format(min_val_loss),
        ])

        with open(os.path.join("train_stats.txt"), "w") as f:
            f.write(t_corpus)
            print(t_corpus)

    # Testing
    elif args.test:
        assert os.path.isfile(checkpoint) and os.path.exists(args.path)
        type_ = pathlib.Path(args.path).name
        ds, num_steps, labels = build_dataset(type_, training=False)
        model = mb.build_model(input_size=INPUT_SIZE, d_model=vocab_size+1)
        model.load_weights(checkpoint)
        model.summary()

        start_time = datetime.datetime.now()

        predictions = model.predict(ds, steps=num_steps)

        # CTC decode
        ctc_decode = True
        if ctc_decode:
            predicts, probabilities = [], []
            x_test = np.array(predictions)
            x_test_len = [MAX_LABEL_LENGTH for _ in range(len(x_test))]

            decode, log = K.ctc_decode(x_test,
                                    x_test_len,
                                    greedy=True,
                                    beam_width=10,
                                    top_paths=1)

            probabilities = [np.exp(x) for x in log]
            predicts = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts = np.swapaxes(predicts, 0, 1)
            predicts = [labels_to_text(label[0]) for label in predicts]
        else:
            predicts = decode_batch(predictions)

        total_time = datetime.datetime.now() - start_time
        print(predicts[:10])
        print(labels[:10])
#         predicts = [x.replace(PAD_TK, "") for x in predicts]
        prediction_file = os.path.join('.', 'predictions_{}.txt'.format(type_))

        with open(prediction_file, "w") as f:
            for pd, gt in zip(predicts, labels):
                f.write("Y {}\nP {}\n".format(gt, pd))

        evaluate = evaluation.ocr_metrics(predicts=predicts,
                                          ground_truth=labels,
                                          norm_accentuation=False,
                                          norm_punctuation=False)

        e_corpus = "\n".join([
            "Total test images:    {}".format(len(labels)),
            "Total time:           {}".format(total_time),
            "Metrics:",
            "Character Error Rate: {}".format(evaluate[0]),
            "Word Error Rate:      {}".format(evaluate[1]),
            "Sequence Error Rate:  {}".format(evaluate[2]),
        ])

        with open("evaluate_stats.txt", "w") as lg:
            lg.write(e_corpus)
            print(e_corpus)
