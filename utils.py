import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.constraints as constraints
from tensorflow.keras.layers import Activation



from sklearn.metrics import confusion_matrix,accuracy_score,balanced_accuracy_score
import numpy as np
import pickle


def save_history(name,data):
    with open(f'{name}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_model(model, train_x, train_y, validation_x, validation_y, epochs=100, batch_size=16,verbose=1):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

    history = model.fit(train_x, train_y, epochs=epochs,batch_size=batch_size, 
                    validation_data=(validation_x, validation_y),shuffle=True,callbacks=[callback], verbose=verbose)

    return model, history.history

def print_class_counts(labels):
    print([np.count_nonzero(np.array(labels)==i) for i in range(len(set(labels)))])


def model_stats(model, gt_x, gt_y):
    stats = {}
    train_probs = model.predict(gt_x,batch_size=64,verbose=0)
    pred_y = [np.argmax(x) for x in train_probs]

    conf_matrix = confusion_matrix(gt_y, pred_y)

    stats['confusion_matrix'] = conf_matrix

    stats['balanced_accuracy_score'] = balanced_accuracy_score(gt_y, pred_y)
    stats['accuracy_score'] = accuracy_score(gt_y, pred_y)
    stats['categorical_accuracy_scores'] = conf_matrix.diagonal()/conf_matrix.sum(axis=1)
    return stats


def get_data():
    # Load data
    with open('aimotive_cifar_application_test.pkl', 'rb') as f:
        data = pickle.load(f)

    output_data = {'train' : {},
                   'pool' : {},
                   'test' : {}}

    output_data['train'] = data['initial_data']
    output_data['pool'] = data['pool']
    output_data['test'] = data['test_set']

    return output_data

def prepare_data(data_x, data_y):
    data_x = np.array(data_x).astype('float32') 
    data_x = data_x / 255.0
    data_y = np.eye(10)[data_y] 

    return data_x, data_y

def get_kaggle_model(num_classes=10, seed=42):
    # source https://www.kaggle.com/code/ektasharma/simple-cifar10-cnn-keras-code-with-88-accuracy
    tf.random.set_seed(seed)

    model = Sequential()

    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax')) 

    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])


    return model

