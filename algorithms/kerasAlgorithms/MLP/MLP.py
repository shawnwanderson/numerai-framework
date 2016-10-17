import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

class MLP(Algorithm):

    def __init__():

    def load_training_data(datasets_location):
        #data for training
        training_data = pd.read_csv(datasets_location + 'numerai_training_data.csv')

        n_examples = training_data.shape[0]
        n_features = training_data.shape[1] - 1

        #check class balance
        num_positive = training_data.target.sum()
        num_neg = n_examples - num_positive
        class_balance = float(num_positive) / float(n_examples)
        if  (0.4 < class_balance) or (class_balance < 0.6):
            print("warning: class balance is skewed:")

        y_train = training_data.target.values.astype('float32')
        X_train = training_data.drop(["target"], axis=1).values.astype('float32')
        return (X_train, y_train)

    def compile_model(datasets_location):
        """
        compile and save the model
        """
        X_train, y_train = training_data(datasets_location)

        model = Sequential()
        model.add(Dense(512, input_shape=(n_features,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.summary()

        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(),
                      metrics=['accuracy'])

    def get_model():


    def run_experiment(datasets_location):
        """ 
        Run cross-validation 5 times, withholding 20% of data for validation
        return binary_crossentropy aka logloss averaged over trials
        """
        batch_size = 128
        nb_classes = 10
        nb_epoch = 20

        hist = model.fit(X_train, y_train, validation_split=0.2,  batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)


    def loglikely(datasets_location):
        with open(datasets_location + 'score.txt', 'r') as f:
            score = f.read()
            return score

    def train_all_data(datasets_location):
        """
        Train a model for tournament predection - trained on all train data
        save weights and model
        """
        #data for submitting predictions
        tournament_data = pd.read_csv(datasets_location + 'numerai_tournament_data.csv')
        #data for training
        training_data = pd.read_csv(datasets_location + 'numerai_training_data.csv')

        n_examples = training_data.shape[0]
        n_features = training_data.shape[1] - 1

        #check class balance
        num_positive = training_data.target.sum()
        num_neg = n_examples - num_positive

        y_train = training_data.target.values.astype('float32')
        X_train = training_data.drop(["target"], axis=1).values.astype('float32')

        model = Sequential()
        model.add(Dense(512, input_shape=(n_features,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.summary()

        model.compile(loss='binary_crossentropy',
                      optimizer=SGD(),
                      metrics=['accuracy'])

        batch_size = 128
        nb_classes = 10
        nb_epoch = 20

        hist = model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
        # serialize model to JSON
        model_json = model.to_json()
        with open(datasets_location + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(datasets_location + "model.h5")
        print("Saved model to disk") 
