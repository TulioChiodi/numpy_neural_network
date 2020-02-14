import matplotlib.pyplot as plt
from IPython.display import Audio
import random
import numpy as np
from sklearn.model_selection import train_test_split


def train_test_splitting(label_lst, spec_flatten_lst):
    Y = label_lst.reshape(1,-1).T
    X = spec_flatten_lst
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.T
    y_test = y_test.T
    layer_dims = (8064,20,7,5,1)
    
    return x_train, x_test, y_train, y_test, layer_dims

def initialize_parameters(layer_dims):
    """
    input:layer dimensions list

    output: parameters dictionary ('W1','b1',...,'WL','bL')
 
    """
    
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l],1))

    return parameters

def linear_forward(W, A, b):
    """
    input:Weights, Activations (or input if A[0]) and bias
    
    output:cache (for backpropagation adjustments), Z (linear response)
    """

    Z = np.dot(W,A)+b
    
    cache = (W, A, b)
    
    return Z, cache

def sigmoid(Z):
    """
    implementation of sigmoid function

    input: Weights, activation and bias;

    output: sigmoid(), cache
    """
    cache = Z

    return 1/(1+np.exp(-Z)), cache

def relu(Z):
    """
    implementation of rectified linear unit.

    input: Weights, activation, bias,

    output: ReLU(Z), cache.

    """
    
    cache = Z

    return np.maximum(0,Z), cache

def linear_activation_forward(W, A_prev, b, activation):
    """
    input: Weights, previous activation and bias

    output: cache, A resulting activation
    """
    
    Z, linear_cache = linear_forward(W, A_prev, b)
    
    if activation == 'sigmoid':
        A, act_cache = sigmoid(Z)

    if activation == 'relu':
        A, act_cache = relu(Z)

    cache = (linear_cache, act_cache)

    return A, cache

def model_forward(X, parameters):
    """
    model for forward propagation [LINEAR->RELU]*N-1 -> [LINEAR->SIGMOID];

    input: X, parameters;

    output: AL (prediction ) and caches (for backpropagation)

    """
    L = len(parameters)//2
    A_prev = X
    cache = []

    for l in range(1, L):
        A_prev, cache_relu = linear_activation_forward(parameters[f'W{l}'], A_prev, parameters[f'b{l}'], activation='relu')
        cache.append(cache_relu)

    AL, cache_sig = linear_activation_forward(parameters[f'W{L}'], A_prev, parameters[f'b{L}'], activation='sigmoid')

    cache.append(cache_sig)
    return AL, cache

def cost_func(AL, Y):
    """
    Compute cost function with cross-entropy cost

    input: AL (prediction), y (ground truth)

    output: cost
    """

    m = AL.shape[1]
    
    cost = -(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))

    return cost


def linear_backward(dZ, linear_cache):
    """
    backward propagation linear part.

    input: dZ (dCost/dZ), cache (W, A_prev, b);

    output: dA_prev, dW, db.
    """

    W, A_prev, b = linear_cache
    m = A_prev.shape[1]
    
    dW = 1./m * dZ.dot(A_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dW, dA_prev, db

def sigmoid_backward(dA,act_cache):
    """
    calculating dZ = dA*g'(Z) with g = 'sigmoid'

    input: dA, Z(act_cache)

    output: dZ
    """
    Z = act_cache
    a = 1/(1+np.exp(-Z))
    dZ = dA*a*(1-a)

    return dZ

def relu_backward(dA, act_cache):
    """
    calculating dZ = dA*g'(Z) with g = 'ReLU'

    input: dA, Z (act_cache);

    output: dZ
    """
    
    Z = act_cache

    dZ = np.copy(dA)
    
    dZ[Z <= 0] = 0

    return dZ

def linear_activation_backward(dA, cache, activation):
    """
    linear + activation backward functions
    
    input: dA, cache (linear and act cache), activation ('relu' or 'sigmoid')
    
    output: dW, dA_prev, db
    """   
    
    linear_cache, act_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, act_cache)
        dW, dA_prev, db = linear_backward(dZ, linear_cache)
    
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, act_cache)
        dW, dA_prev, db = linear_backward(dZ, linear_cache)

    return dW, dA_prev, db

def model_backwards(AL, Y, cache):
    """
    model for backwards propagation LINEAR->SIGMOID and LINEAR->ReLU
    input: AL, Y, cache;

    output: grads (gradients dW, A_prev, db)
    """

    grads = {}
    L = len(cache)
    m = Y.shape[1]

    dAL = - (np.divide(Y,AL) - np.divide(1-Y, 1-AL))

    current_cache = cache[L-1]
    dW, dA_prev, db = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    grads[f'dW{L}'] = dW 
    grads[f'dA{L-1}']= dA_prev
    grads[f'db{L}'] = db

    for l in reversed(range(1, L)):
        current_cache=cache[l-1]
        dW, dA_prev, db = linear_activation_backward(grads[f'dA{l}'], current_cache, activation='relu')
        grads[f'dW{l}'] = dW
        grads[f'dA{l-1}'] = dA_prev
        grads[f'db{l}'] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2

    for l in range(1, L+1):
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']

    return parameters


def model(X, y, layer_dims, learning_rate, iter_number):
    """
    complete model using helper functions.

    input: input data, input labels, layer dimensions, learning_rate
    
    output: parameters
    """
    plt.rcParams['figure.figsize'] = (5.0, 4.0)
    np.random.seed(42)
    costs = []
    
    parameters = initialize_parameters(layer_dims)

    for i in range (0, iter_number):

        AL, caches = model_forward(X, parameters)

        cost = cost_func(AL, y)

        grads = model_backwards(AL, y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print(f"Cost (iter:{i}): {cost}")
            costs.append(cost)


    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundred)')
    plt.title(f'Cost with learning rate: {learning_rate}')
    plt.show()

    return parameters


def predict(X, y, parameters):

    m = X.shape[1]
    p = np.zeros((1,m))

    pred, _ = model_forward(X, parameters)

    for i in range(0, pred.shape[1]):
        if pred[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print(f'Accuracy:{str(np.sum((p == y)/m))}')

    return p

def plot_mislabeled(classes, X, y, p, shape, signal_list, sample_rate, display_audio):
    """
    Plotting mislabeled spec sounds
    
    inputs: classes, X (x_test), y (y_test), p (pred_test), shape (of the spectrogram), display_audio (True for yes False for no)
    
    output: mislabeled audio indexes
    """
    plt.rcParams['figure.figsize'] = (12.0, 16.0) # set default size of plots
    y_int = [[]]
    y_int[0] = [int(item) for item in y[0]]
    a = p + y_int
    mislabeled_indices = np.asarray(np.where(a == 1))
    num_images = len(mislabeled_indices[0])
    print(f'Number of mislabeled audios: {num_images}')
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(3, num_images, i + 1)
        plt.imshow(X[:,index].reshape(shape), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0][index])] + " \n Class: " + classes[y_int[0][index]])
        if display_audio:
            display(Audio(signal_list[index], rate=sample_rate))
            plt.show()
    return mislabeled_indices[1]
