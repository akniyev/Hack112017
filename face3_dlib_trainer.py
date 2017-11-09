#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#
# This is an example illustrating the use of a binary SVM classifier tool from
# the dlib C++ Library.  In this example, we will create a simple test dataset
# and show how to learn a classifier from it.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#

import dlib
import pickle
import sys
import os
from PyQt5.QtWidgets import *
import numpy as np

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

def trainNet(x, y, c):
    HIDDEN_NEURONS = 100
    WEIGHTDECAY = c
    MOMENTUM = 0.1

    fnn = buildNetwork(402, HIDDEN_NEURONS, 1,
                       outclass=SoftmaxLayer)

    ds = ClassificationDataSet(402, 1, nb_classes=2)
    for xi, yi in zip(x, y):
        ds.addSample(tuple(xi), yi)


    trainer = BackpropTrainer(fnn, dataset=ds, momentum=MOMENTUM,
                              verbose=True, weightdecay=WEIGHTDECAY)

    # for i in range(100):
    #     print(trainer.train())
    #trainer.trainUntilConvergence()
    #return fnn
    svm = dlib.svm_c_trainer_histogram_intersection()
    svm.set_c(c)
    return svm.train(x, y)

def accuracy(classifier, x, y):
    hits = 0
    for sample, label in zip(x, y):
        #print("%3i, %5.2f" % (label, classifier(sample)))
        if label * classifier(sample) > 0.001:
            hits = hits + 1
    return hits / len(y)

def load_data_from_file(dir, file_prefix) -> (int, dlib.array):
    eye1_data_file = os.path.join(dir, file_prefix + '_eye1.txt')
    eye2_data_file = os.path.join(dir, file_prefix + '_eye2.txt')
    nose_data_file = os.path.join(dir, file_prefix + '_nose.txt')
    label_data_file = os.path.join(dir, file_prefix + '_y.txt')

    eye1_array = np.loadtxt(eye1_data_file, dtype=np.int32).flatten()
    eye2_array = np.loadtxt(eye2_data_file, dtype=np.int32).flatten()
    nose_array = np.loadtxt(nose_data_file, dtype=np.int32).flatten()

    training_example_data = dlib.vector(np.concatenate((eye1_array, eye2_array, nose_array)).tolist())
    training_example_label = 1 if np.loadtxt(label_data_file, dtype=np.int32).flatten().tolist()[0] == 1 else -1

    # print(np.loadtxt(label_data_file, dtype=np.int32).flatten().tolist()[0])

    return (training_example_label, training_example_data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dirname = QFileDialog.getExistingDirectory()

    files = []
    for file in os.listdir(dirname):
        if file.endswith('.jpg'):
            files.append(file[:-4])

    x = dlib.vectors()
    y = dlib.array()

    for file_prefix in files:
        label, array = load_data_from_file(dirname, file_prefix)
        x.append(array)
        y.append(label)

    # svm = dlib.svm_c_trainer_radial_basis()
    #svm = dlib.svm_c_trainer_linear()
    svm = dlib.svm_c_trainer_histogram_intersection()
    svm.set_c(10)

    trainInd = int(len(x) * 0.6)
    trainCross = int(len(x) * 0.8)
    xtrain = x[:trainInd]
    ytrain = y[:trainInd]
    xcross = x[trainInd:trainCross]
    ycross = y[trainInd:trainCross]
    xtest = x[trainCross:]
    ytest = y[trainCross:]

    classifier = svm.train(xtrain, ytrain)

    #hits = 0
    #for c in [2**i for i in range(-2, 14)]:
        #fnn = trainNet(xtrain, ytrain, c)
        # print("c=%10.2f; Train acc=%10.2f; Cross acc=%10.2f" % (c, accuracy(lambda x: fnn.activate(x), xtrain, ytrain), accuracy(lambda x: fnn.activate(x), xcross, ycross)))
        # print("Cross: c=%i, acc=%5.2f" % (c, accuracy(h, xcross, ycross)))



    example = dlib.vector(np.loadtxt('r.txt'))
    print(classifier(example))

    # classifier models can also be pickled in the same was as any other python object.
    classifier = trainNet(x, y, 10)
    with open('saved_model.pickle', 'wb') as handle:
        pickle.dump(classifier, handle)

    sys.exit(app.exec_())

