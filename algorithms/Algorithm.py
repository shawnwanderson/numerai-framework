import pandas as pd
import numpy as np

class Algorithm:

    def __init__(_datasets_location):
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

