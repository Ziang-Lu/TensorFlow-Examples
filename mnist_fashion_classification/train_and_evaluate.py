#!usr/env/bin python3
# -*- coding: utf-8 -*-

"""
Module to construct, train and evaluate the model.
"""

__author__ = 'Ziang Lu'

import pickle

import tensorflow as tf
from tensorflow import keras

from data_preparation import (
    TRAIN_IMGS_FILENAME, TRAIN_LABELS_FILENAME, TEST_IMGS_FILENAME,
    TEST_LABELS_FILENAME
)


def _construct_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    return model


def _retrieve_data() -> tuple:
    with open(TRAIN_IMGS_FILENAME, 'rb') as f:
        train_imgs = pickle.load(f)
    with open(TRAIN_LABELS_FILENAME, 'rb') as f:
        train_labels = pickle.load(f)
    with open(TEST_IMGS_FILENAME, 'rb') as f:
        test_imgs = pickle.load(f)
    with open(TEST_LABELS_FILENAME, 'rb') as f:
        test_labels = pickle.load(f)
    return train_imgs, train_labels, test_imgs, test_labels


def main():
    nn_model = _construct_model()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_imgs, train_labels, test_imgs, test_labels = _retrieve_data()
    model.fit(train_imgs, train_labels, epochs=5)
    _, test_acc = model.evaluate(test_imgs, test_labels)
    print(f'Test accuracy = {test_acc}')


if __name__ == '__main__':
    main()
