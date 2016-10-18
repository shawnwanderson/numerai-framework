import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import datetime

now = datetime.datetime.now()
month = now.strftime("%B")
default_data_location = "../numerai_datasets/" + month + "/"

class Algorithm(object):

    def __init__(self, _datasets_location=default_data_location):
        self.model_location = "../models/" + self.name + "/"
        self.datasets_location = _datasets_location 

    def load_training_data(self):
        #data for training
        training_data = pd.read_csv(self.datasets_location + 'numerai_training_data.csv')

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

class KerasAlgorithm(Algorithm):
    """Base class for models implementing Keras"""

    def compile_model(self):
        """
        compile and save the model
        """
        pass

    def test_model(self, batch_size=128, np_epoch=20):
        """ 
        Run cross-validation on a 20% holdout 
        return binary_crossentropy aka logloss averaged over trials
        """
        model = self.get_model()
        hist = model.fit(X_train, y_train, validation_split=0.2,  batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
        print(hist.history)
        self.save_history(hist.history)

    def return_fully_trained(self, batch_size=128, nb_epoch=20):
        """
        Train a model for tournament predection - trained on all train data
        save weights and model
        """
        try:
            model = load_model("fully_trained.json")
        except:
            model = self.get_model()
            hist = model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
            #save the model
            self.serialize_model(model, "fully_trained.json")
            #save the weights
            self.save_weights(model, "fully_trained.hd5")
        finally:
            return model

    def serialize_model(self, model, name):
	"""model is saved in model.json,"""
        # serialize model to JSON
        model_json = model.to_json()
        with (self.model_location + name, "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk") 

    def load_model(self, name):
	with open(self.model_location + name, "r") as json_file
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            return loaded_model


    def save_weights(self, model, name="model.hd5"):
        """ weight in model.h5"""
        # serialize weights to HDF5
        model.save_weights(self.model_location + "model.h5")
        print("Saved weights to disk") 

    def return_score(self):
        """cache the logloss/binary cross-entropy"""
        with open(self.model_location + 'score.txt', 'r') as f:
            score = f.read()
            return score

    def get_compiled_model(self):
	"""try to load cached model. Otherwise compile the model"""
        try:
            load_model("compiled_model.json")
        except(IOError):
            print("model not cached")
            while True:
                a = raw_input("compile model and save to " + self.model_location + "/compiled_model.json ? Enter y/n to continue") 
                if a == 'y':
                    self.compile_model()
                    break
                elif a=='n':
                    break
                else:
                    print("Enter y/n")




