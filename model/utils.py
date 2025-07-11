import numpy as np
import tensorflow as tf
import h5py
import math
import os

def load_dataset():
    base_path = os.path.join(os.path.dirname(__file__), '../datasets')
    
    train_dataset = h5py.File("C:/Users/Devavrata/fashion_trend_ai/model/datasets/train_signs.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("C:/Users/Devavrata/fashion_trend_ai/model/datasets/test_signs.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def random_mini_batches(x,y,mini_batch_size=64, seed=0):
    m = x.shape[0]
    mini_batches=[]
    np.random,seed(seed)

    permutation = list(np.random.permutation(0))
    shuffled_x = x[permutation,:,:,:]
    shuffled_y = y[permutation,:]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[k * mini_batch_size : k*mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_y = shuffled_y[k*mini_batch_size:k*mini_batch_size+mini_batch_size,:]
        mini_batch = (mini_batch_x,mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[num_complete_minibatches*mini_batch_size: m, :,:,:]
        mini_batch_y = shuffled_y[num_complete_minibatches*mini_batch_size:m,:]
        mini_batch = (mini_batch_x,mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches

def convert_to_one_hot(y, c):
    y = np.eye(c)[y.reshape(-1)].T
    return y

def forward_propagation_for_predict(x, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
    Z1 = tf.add(tf.matmul(W1,x),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)

    return Z3

def predict(x, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1":W1,
              "b1":b1,
              "W2":W2,
              "b2":b2,
              "W3":W3,
              "b3":b3}
    
    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x:x})

    return prediction