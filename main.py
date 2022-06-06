import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Activation, Flatten, Dropout, Softmax, AvgPool2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
# from sklearn.metrics import confusion_matrix


def download_playlist():
    playlist_dl_dir = "D:\\uni\\deepLearning\\dataset_2"
    dl_string = f"spotify_dl -o {playlist_dl_dir}"
    pl_link = input("Enter playlist link: ")
    os.system(dl_string+f" -l {pl_link}")


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
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dropout(0.2))
    # model.add(Dense(50))
    # model.add(Activation('sigmoid'))
    model.add(Dense(20))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_size))
    model.add(Softmax())

    return model, models_no


def train_model(model, x, y, models_no, folder_no, lr=0.001):
    model_filepath = f"D:\\uni\\deepLearning\\models\\models_{folder_no}\\{models_no}\\"
    model_checkpoint_filepath = model_filepath + '{epoch:02d}-{loss:.7f}-{val_loss:.7f}.hdf5'
    callbacks = [EarlyStopping(monitor='val_loss', patience=150),
                 ModelCheckpoint(model_checkpoint_filepath, monitor='loss', save_best_only=True, mode='min')]
    optimizer = SGD(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x, y, validation_split=0.2, epochs=1000, callbacks=callbacks, batch_size=8)
    return history


def load_n_predict(model_path: str, x, y):
    model = keras.models.load_model(model_path)
    predictions = model.predict(x)
    return predictions, y


def main():
    models_set = [
        [16, 8], [32, 16], [64, 32],
        [16, 8, 4], [32, 16, 8], [64, 32, 16],
    ]

    pooling_set = [
        [3, 3], [3, 3], [3, 3],
        [3, 2, 2], [3, 2, 2], [3, 2, 2],
    ]
    folder_no = 6

    general_dir = "D:\\uni\\deepLearning\\dataset_2"
    x_train, x_test, y_train, y_test = create_dataset(general_dir)
    for count, s in enumerate(models_set):
        mdl, mdl_no = establish_model(s, pooling_set[count], 3)
        history = train_model(mdl, x_train, y_train, mdl_no, folder_no, lr=0.0001)
        model_filepath = f"D:\\uni\\deepLearning\\models\\models_{folder_no}\\{mdl_no}\\"
        train_acc = mdl.evaluate(x_train, y_train)
        test_acc = mdl.evaluate(x_test, y_test)
        predictions = mdl.predict(x_test)
        # conf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
        with open(model_filepath+'eval.txt', 'w', encoding='utf-8') as f:
            f.write(f"Train acc: {train_acc}\n"
                    f"Test acc: {test_acc}\n"
                    f"y_test:\n{y_test}\n\n"
                    f"predictions:\n{predictions}")
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.legend()
        plt.savefig(f'{model_filepath}val_graph.png')
        plt.close()
    print("fin")


def do_statistics(pred, actual_y):
    pass


if __name__ == "__main__":
    # main()
    general_dir = "D:\\uni\\deepLearning\\dataset_2"
    x_train, x_test, y_train, y_test = create_dataset(general_dir)
    model_folder_dir = 'D:\\uni\\deepLearning\\models\\models_4'
    model_dir = 'model_32_16\\311-0.0950032-0.1710008.hdf5'
    pred, actual_y = load_n_predict(f'{model_folder_dir}\\{model_dir}', x_test, y_test)
    with open('predicions.txt', 'w', encoding='utf-8') as f:
        f.write(f'predictions:\n'
                f'{pred}\n\n'
                f'actual y:\n'
                f'{actual_y}')
    pass
