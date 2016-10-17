from keras.models import Sequential

class KerasAlgorithm(Algorithm):
    """Base class for models implementing Keras"""

    def compile_model(self):
        """
        compile and save the model
        """
        pass

    def run_experiment(self):
        pass

    def train_all_data(self):
        pass


    def serialize_model(self, model):
	"""model is saved in model.json,"""
        # serialize model to JSON
        model_json = model.to_json()
        with open(datasets_location + "model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk") 

    def save_weights(self, model):
        """ weight in model.h5"""
        # serialize weights to HDF5
        model.save_weights(datasets_location + "model.h5")
        print("Saved weights to disk") 

    def loglikely(self):
        with open(self.datasets_location + 'score.txt', 'r') as f:
            score = f.read()
            return score

    def get_model(self):
	"""try to load cached model. Otherwise compile the model"""
        try:
            json_file = open(self.datasets_location + "model.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            return loaded_model
        except(FileNotFoundError):
            print("model not cached")
            while True:
                a = input("compile model and save to " + self.datasets_location + "/model.json ? Enter y/n to continue") 
                if a == "y":
                    self.compile_model()
                    break
                elif a=='n':
                    break
                else:
                    print("Enter y/n")




