import numpy as np


# LOAD THE LIST OF CLASSES
def get_classes(file="data/speech-commands/classes.txt"):
    words = open(file).read().split()
    return words


def load_data(train_file="data/speech-commands/train.npz", test_file="data/speech-commands/test.npz"):
    train_data = np.load(train_file)
    Xtrain = train_data["arr_0"]
    Ytrain = train_data["arr_1"]
    test_data = np.load(test_file)
    Xtest = test_data["arr_0"]
    Ytest = test_data["arr_1"]
    return Xtrain, Ytrain, Xtest, Ytest

# SHOW A SAMPLE IMAGE
# image = Xtrain[8, :].reshape(20, 80)
# plt.imshow(image)
# plt.colorbar()
# plt.show()
# counters = np.bincount(Ytrain)
# for i in range(35):
#     print(words[i], "\t", counters[i])


# MEAN/VARIANCE NORMALIZATION

def mean_variance_normalization(x_train, x_test):
    mu = x_train.mean(0)
    std = x_train.std(0)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std
    return x_train, x_test


def minmax_normalization(Xtrain, Xtest):
    xmin = Xtrain .min(0)
    xmax = Xtrain .max(0)
    Xtrain = (Xtrain - xmin) / (xmax - xmin)
    Xtest = (Xtest - xmin) / (xmax - xmin)
    return Xtrain, Xtest


def maxabs_normalization(Xtrain, Xtest):
    amax = np .abs(Xtrain) .max(0)
    Xtrain = Xtrain / amax
    Xtest = Xtest / amax
    return Xtrain, Xtest


def whitening(Xtrain, Xtest):
    mu = Xtrain . mean(0)
    sigma = np . cov(Xtrain . T)
    evals, evecs = np . linalg . eigh(sigma)
    w = evecs / np . sqrt(evals)
    Xtrain = (Xtrain - mu) @ w
    Xtest = (Xtest - mu) @ w
    return Xtrain, Xtest


# image = mu.reshape(20, 80)
# plt.imshow(image)
# plt.colorbar()
# plt.title("Mean")
# plt.figure()
# image = std.reshape(20, 80)
# plt.imshow(image)
# plt.colorbar()
# plt.title("Stddev")
# plt.show()

# COMPUTE THE ACCURACY
def accuracy(net, X, Y):
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100
