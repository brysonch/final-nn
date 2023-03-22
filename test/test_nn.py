import numpy as np
import numpy.testing as testing
from nn import io, nn, preprocess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

nn_arch = [{'input_dim': 8, 'output_dim': 4, 'activation': 'relu'}, {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}]
lr = 0.1
seed = 42
batch_size = 10
epochs = 100
loss_function_mse = 'mse'
loss_function_bce = 'bce'

nn_test = nn.NeuralNetwork(nn_arch=nn_arch, lr=lr, seed=seed, batch_size=batch_size, epochs=epochs, loss_function=loss_function_mse)

def test_single_forward():
    A_test, Z_test = nn_test._single_forward(nn_test._param_dict['W2'], nn_test._param_dict['b2'], np.random.randint(2, size=len(nn_test._param_dict['W1'])), nn_arch[0]['activation'])
    
    testing.assert_allclose(A_test, np.array([[0, 0]]), rtol=1e-4, atol=0)
    testing.assert_allclose(Z_test, np.array([[-0.47663782, -0.06641242]]), rtol=1e-4, atol=0)

def test_forward():
    X_test = np.arange(1, 33).reshape(4,8)
    A_out, cache = nn_test.forward(X_test)

    testing.assert_allclose(A_out, np.array([[0.47615291, 0.52241666],
        [0.49338545, 0.58836627],
        [0.51179431, 0.64967263],
        [0.53017123, 0.70640456]]), rtol=1e-4, atol=0)

    testing.assert_allclose(cache['A1'], np.array([[ 2.28602523, -0., -0.,  0.23593534],
         [ 5.8120456 , -0., -0., -0.],
         [ 9.33806596, -0., -0., -0.],
         [12.86408633, -0., -0., -0.]]), rtol=1e-4, atol=0)

    testing.assert_allclose(cache['Z1'], np.array([[  2.28602523,  -3.05378325,  -1.28867806,   0.23593534],
         [  5.8120456 ,  -6.90545135,  -3.79768532,  -0.44447791],
         [  9.33806596, -10.75711945,  -6.30669258,  -1.12489116],
         [ 12.86408633, -14.60878755,  -8.81569985,  -1.80530441]]), rtol=1e-4, atol=0)

def test_single_backprop():
    W_curr = np.arange(100, 132).reshape(4,8)
    b_curr = np.arange(61, 65).reshape(4,1)
    Z_curr = np.arange(85, 125).reshape(10,4)
    A_prev = np.arange(22, 102).reshape(10,8)
    dA_curr = 2

    dA_prev, dW_curr, db_curr = nn_test._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, 'relu')
    
    testing.assert_allclose(dA_prev, np.array([[896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952],
        [896, 904, 912, 920, 928, 936, 944, 952]]), rtol=1e-4, atol=0)

    testing.assert_allclose(dW_curr, np.array([[1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300],
        [1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300],
        [1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300],
        [1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300]]), rtol=1e-4, atol=0)

    testing.assert_allclose(db_curr, np.array([[20],
        [20],
        [20],
        [20]]), rtol=1e-4, atol=0)

def test_predict():
    X_test = np.arange(1, 33).reshape(4,8)
    pred = nn_test.predict(X_test)

    testing.assert_allclose(pred, np.array([[0.47615291, 0.52241666],
       [0.49338545, 0.58836627],
       [0.51179431, 0.64967263],
       [0.53017123, 0.70640456]]), rtol=1e-4, atol=0)

def test_binary_cross_entropy():
    y = np.array([1, 1, 0, 1, 1])
    y_hat = np.array([0.8, 0.4, 0.1, 0.7, 0.2])

    testing.assert_allclose(log_loss(y, y_hat), nn_test._binary_cross_entropy(y, y_hat), rtol=1e-4, atol=0)

def test_binary_cross_entropy_backprop():
    y = np.array([1, 1, 0, 1, 1])
    y_hat = np.array([0.8, 0.4, 0.1, 0.7, 0.2])

    assert nn_test._binary_cross_entropy_backprop(y, y_hat) - (-1.8134920634920633) < 1e-05

def test_mean_squared_error():
    y = np.array([1.1, 3.4, 8.9, 5.6, 7.2])
    y_hat = np.array([1.4, 3.2, 8.8, 5.7, 7.7])

    testing.assert_allclose(mean_squared_error(y, y_hat), nn_test._mean_squared_error(y, y_hat), rtol=1e-4, atol=0)

def test_mean_squared_error_backprop():
    y = np.array([1.1, 3.4, 8.9, 5.6, 7.2])
    y_hat = np.array([1.4, 3.2, 8.8, 5.7, 7.7])

    assert nn_test._mean_squared_error_backprop(y, y_hat) - 0.24000000000000038 < 1e-05

def test_sample_seqs():
    test_seq = ['AGTACAGAT', 'GCAGTCCGG', 'TTAGATTGC', 'GATCGGATC', 'ATCGTGTCA', 'CGGACTATT']
    test_label = [1, 0, 0, 1, 0, 0]

    classes = sum(test_label)
    if classes < len(test_label) / 2:
        larger_class = test_label.count(0)
    else: 
        larger_class = test_label.count(1)

    d = {}
    for idx, s in enumerate(test_seq):
        d[s] = test_label[idx]

    processed_seqs, processed_labels = preprocess.sample_seqs(test_seq, test_label)

    assert len(processed_seqs) == len(processed_labels) == larger_class * 2
    
    for idx, s in enumerate(processed_seqs):
        assert s in d
        assert d[s] == processed_labels[idx]

def test_one_hot_encode_seqs():
    test_seq = ['AGTACAGAT', 'GCAGTCCGG']

    assert (preprocess.one_hot_encode_seqs(test_seq)[1] == np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])).all()
