#!usr/env/bin python3
# -*- coding: utf-8 -*-

"""
Data preparation module.
"""

__author__ = 'Ziang Lu'

import pickle

from tensorflow import keras

# CLASS_NAMES = [
#     'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
#     'Sneaker', 'Bag', 'Ankle boot'
# ]
TRAIN_IMGS_FILENAME = 'train_imgs'
TRAIN_LABELS_FILENAME = 'train_labels'
TEST_IMGS_FILENAME = 'test_imgs'
TEST_LABELS_FILENAME = 'test_labels'


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_imgs, train_labels), (test_imgs, test_labels) = \
        fashion_mnist.load_data()

    # train_imgs: 60000 x 28 x 28
    # train_labels: 60000 x 1
    # test_imgs: 10000 x 28 x 28
    # test_labels: 10000 x 1

    # Preprocess the data
    # Scale the data to a range [0, 1)
    train_imgs = train_imgs / 255.0
    test_imgs = test_imgs / 255.0

    # Pickle the date
    with open(TRAIN_IMGS_FILENAME, 'wb') as f:
        pickle.dump(train_imgs, f)
    with open(TRAIN_LABELS_FILENAME, 'wb') as f:
        pickle.dump(train_labels, f)
    with open(TEST_IMGS_FILENAME, 'wb') as f:
        pickle.dump(test_imgs, f)
    with open(TEST_LABELS_FILENAME, 'wb') as f:
        pickle.dump(test_labels, f)


if __name__ == '__main__':
    main()
