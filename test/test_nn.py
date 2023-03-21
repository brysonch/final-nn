import numpy as np
from nn import io, nn, preprocess
from sklearn.metrics import mean_squared_error

nn_arch = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
lr = 0.1
seed = 42
batch_size = 10
epochs = 100
loss_function_mse = 'mse' 

nn_test = NeuralNetwork(nn_arch=nn_arch, lr=lr, seed=seed, batch_size=batch_size, epochs=epochs, loss_function=loss_function_mse)

def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    y = [1.1, 3.4, 8.9, 5.6, 7.2]
    y_hat = [1.4, 3.2, 8.8, 5.7, 7.7]

def test_binary_cross_entropy_backprop():
    y = [1.1, 3.4, 8.9, 5.6, 7.2]
    y_hat = [1.4, 3.2, 8.8, 5.7, 7.7]

def test_mean_squared_error():
    y = [1.1, 3.4, 8.9, 5.6, 7.2]
    y_hat = [1.4, 3.2, 8.8, 5.7, 7.7]

    assert np.allclose(mean_squared_error(y, y_hat), nn_test._mean_squared_error(y, y_hat))

def test_mean_squared_error_backprop():
    y = [1.1, 3.4, 8.9, 5.6, 7.2]
    y_hat = [1.4, 3.2, 8.8, 5.7, 7.7]

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    test_seq = 'AGTACAGAT'

    #assert one_hot_encode_seqs(test_seq) == 