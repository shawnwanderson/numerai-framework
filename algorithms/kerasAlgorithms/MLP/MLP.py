import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

class MLP(KerasAlgorithm):

    def compile_model(self):
        """
        compile and save the model
        """
        X_train, y_train = self.load_training_data()

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

        # serialize model to JSON
        self.serialize_model(model)


    def run_experiment(self):
        """ 
        Run cross-validation on a 20% holdout 
        return binary_crossentropy aka logloss averaged over trials
        """
        model = self.get_model()

        batch_size = 128
        nb_classes = 10
        nb_epoch = 20

        hist = model.fit(X_train, y_train, validation_split=0.2,  batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)


    def train_all_data(self):
        """
        Train a model for tournament predection - trained on all train data
        save weights and model
        """
        model = self.get_model()

        batch_size = 128
        nb_classes = 10
        nb_epoch = 20

        hist = model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)

        #save the weights
        self.save_weights(model)

