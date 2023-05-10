import numpy as np
import pvml
import matplotlib.pyplot as plt


def get_network(file="mlp.npz"):
    net = pvml.MLP.load(file)
    return net


def show_weights(net, words):
    w = net.weights[0]
    maxval = np.abs(w).max()
    plt.figure(figsize=(20, 10))
    for klass in range(35):
        plt.subplot(5, 7, klass + 1)
        plt.imshow(w[:, klass].reshape(20, 80), vmin=-
                   maxval, vmax=maxval, cmap="seismic")
        plt.title(words[klass])
    plt.show()


def make_confusion_matrix(predictions, lables):
    cmat = np.zeros((35, 35))
    for i in range(predictions.size):
        cmat[lables[i], predictions[i]] += 1
    return cmat


def display_confusion_matrix(cmat, words):
    print(" "*10, end="")
    for j in range(35):
        print(f"{words[j][:4]:4}", end="")
    print()
    for i in range(35):
        print(f"{words[i]:10}", end="")
        for j in range(35):
            val = int(cmat[i, j])
            print(f"{val:4d}", end="")
        print()

# DISPLAY THE CONFUSION MATRIX


def show_confusion_matrix(Y, predictions, words):
    classes = Y.max() + 1
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        cm[klass, :] = 100 * counts / max(1, counts.sum())
    plt.figure(3, figsize=(20, 20))
    plt.clf()
    plt.xticks(range(classes), words, rotation=45)
    plt.yticks(range(classes), words)
    plt.imshow(cm, vmin=0, vmax=100, cmap=plt.cm.Reds)
    for i in range(classes):
        for j in range(classes):
            txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
            col = ("black" if cm[i, j] < 75 else "white")
            plt.text(j - 0.25, i, txt, color=col)
    plt.title("Confusion matrix")
