from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from BaseAlgo import KerasAlgorithm

class MultiLayerPerceptron(KerasAlgorithm):
    name =  "MultiLayerPerceptron"
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()

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
        self.serialize_model(model, "compiled_model.json")

