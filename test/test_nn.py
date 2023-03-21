import numpy as np
from nn import io, nn, preprocess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
#from tf.keras.losses.BinaryCrossentropy

nn_arch = [{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation:': 'sigmoid'}]
lr = 0.1
seed = 42
batch_size = 10
epochs = 100
loss_function_mse = 'mse'
loss_function_bce = 'bce'

nn_test = nn.NeuralNetwork(nn_arch=nn_arch, lr=lr, seed=seed, batch_size=batch_size, epochs=epochs, loss_function=loss_function_mse)

def test_single_forward():
    A_test, Z_test = nn_test._single_forward(nn_test._param_dict['W2'], nn_test._param_dict['b2'], np.random.randint(2, size=len(nn_test._param_dict['W1'])), nn_arch[0]['activation'])
    
    assert (A_test == np.array([[0, 0]])).all()
    assert (Z_test == np.array([[-0.1478522 , -0.07198442]])).all()

def test_forward():
    nn_test.forward()

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    y = np.array([0.8, 0.4, 0.1, 0.7, 0.2])
    y_hat = np.array([1, 1, 0, 1, 1])

    assert nn_test._binary_cross_entropy(y, y_hat) - 4.605164186018091 <= 1e-05 

def test_binary_cross_entropy_backprop():
    y = np.array([1.1, 3.4, 8.9, 5.6, 7.2])
    y_hat = np.array([1.4, 3.2, 8.8, 5.7, 7.7])

def test_mean_squared_error():
    y = np.array([1.1, 3.4, 8.9, 5.6, 7.2])
    y_hat = np.array([1.4, 3.2, 8.8, 5.7, 7.7])

    assert mean_squared_error(y, y_hat) == nn_test._mean_squared_error(y, y_hat)

def test_mean_squared_error_backprop():
    y = np.array([1.1, 3.4, 8.9, 5.6, 7.2])
    y_hat = np.array([1.4, 3.2, 8.8, 5.7, 7.7])

    assert nn_test._mean_squared_error_backprop(y, y_hat) - 0.24000000000000038 <= 1e-05

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    test_seq = ['AGTACAGAT', 'GCAGTCCGG']

    assert (preprocess.one_hot_encode_seqs(test_seq)[1] == np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])).all()
