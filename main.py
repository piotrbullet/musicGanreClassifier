import typing
import subprocess
import configparser
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Activation, Flatten, SpatialDropout2D, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def download_playlist():
    config = configparser.ConfigParser()
    config.read('config.ini')
    playlist_dl_dir = config['MAIN']['playlist_dl_dir']
    dl_string = f"spotify_dl -o {playlist_dl_dir}"
    pl_link = input("Enter playlist link: ")
    os.system(dl_string+f" -l {pl_link}")


def shorten_audio(folder_dir: str):
    files = [f for f in os.listdir(folder_dir)]


def produce_spectrograms(folder_dir: str):
    files = [f for f in os.listdir(folder_dir)]


def create_dataset(folder_dir):
    genre_list = [g for g in os.listdir(folder_dir) if os.path.isdir(os.path.join(folder_dir, g))]
    first_assignment = False
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for count, g in enumerate(genre_list):
        train_dir = os.path.join(folder_dir, g, 'spectrogram', 'train')
        test_dir = os.path.join(folder_dir, g, 'spectrogram', 'test')
        train_spectrogram_paths = [os.path.join(train_dir, s) for s in os.listdir(train_dir) if s.endswith('.png')]
        test_spectrogram_paths = [os.path.join(test_dir, s) for s in os.listdir(test_dir) if s.endswith('.png')]
        for count2, f in enumerate(train_spectrogram_paths):
            y_def = np.array([0, 0, 0])
            y_def[count] = 1
            img_array = img_to_array(load_img(f, color_mode="grayscale"))
            x_train.append(img_array)
            y_train.append(y_def)
        for count2, f in enumerate(test_spectrogram_paths):
            y_def = np.array([0, 0, 0])
            y_def[count] = 1
            img_array = img_to_array(load_img(f, color_mode="grayscale"))
            x_test.append(img_array)
            y_test.append(y_def)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test


def establish_model(unit_numbers: list, pooling_sizes: list, output_size):
    models_no = "model"
    model = keras.Sequential()

    for count, n in enumerate(unit_numbers):
        models_no += f"_{n}"

        if count == 0:
            model.add(Conv2D(n, pooling_sizes[count], input_shape=(200, 500, 1)))
        else:
            model.add(Conv2D(n, pooling_sizes[count]))
        model.add(MaxPool2D(pool_size=pooling_sizes[count]))
        model.add(Dropout(0.2))

    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Activation('linear'))
    model.add(Dense(output_size))

    return model, models_no


def train_model(model, x, y, models_no, lr=0.001):
    model_filepath = "models_2/{}/".format(models_no)
    model_checkpoint_filepath = model_filepath + '{epoch:02d}-{loss:.7f}-{val_loss:.7f}.hdf5'
    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(model_checkpoint_filepath, monitor='loss', save_best_only=True, mode='min')]
    tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(x, y, validation_split=0.2, epochs=1000, callbacks=callbacks, batch_size=8)


def main():
    models_set = [[32, 16], [64, 32], [128, 64],
                  [32, 16, 8], [64, 32, 16], [128, 64, 32],
                  [32, 16, 8, 4], [64, 32, 16, 8], [128, 64, 32, 16]]

    pooling_set = [[4, 4], [4, 4], [4, 4],
                   [4, 2, 2], [4, 2, 2], [4, 2, 2],
                   [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]

    general_dir = "D:\\uni\\deepLearning\\musicDataset"
    x_train, x_test, y_train, y_test = create_dataset(general_dir)
    for count, s in enumerate(models_set):
        mdl, mdl_no = establish_model(s, pooling_set[count], 3)
        train_model(mdl, x_train, y_train, mdl_no, lr=0.0001)
    # mdl, mdl_no = establish_model([128, 64], [4, 4], 3)
    # train_model(mdl, x_train, y_train, mdl_no)
    print("fin")


if __name__ == "__main__":
    main()
