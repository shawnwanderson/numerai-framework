from algorithms.MLP import MultiLayerPerceptron as mlp


def compile():
    m = mlp()
    m.compile_model()


def test_model():
    m = mlp()
    m.test_model()

def return_fully_trained():
    m = mlp()
    m.return_fully_trained()
