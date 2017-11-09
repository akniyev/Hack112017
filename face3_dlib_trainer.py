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

def load_data_from_file(dir, file_prefix) -> (int, dlib.array):
    eye1_data_file = os.path.join(dir, file_prefix + '_eye1.txt')
    eye2_data_file = os.path.join(dir, file_prefix + '_eye2.txt')
    nose_data_file = os.path.join(dir, file_prefix + '_nose.txt')
    label_data_file = os.path.join(dir, file_prefix + '_y.txt')

    eye1_array = np.loadtxt(eye1_data_file, dtype=np.int32).flatten()
    eye2_array = np.loadtxt(eye2_data_file, dtype=np.int32).flatten()
    nose_array = np.loadtxt(nose_data_file, dtype=np.int32).flatten()

    training_example_data = dlib.vector(np.concatenate((eye1_array, eye2_array, nose_array)).tolist())
    training_example_label = 1 if np.loadtxt(label_data_file, dtype=np.int32).flatten().tolist()[0] == 0 else -1

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

    svm = dlib.svm_c_trainer_radial_basis()
    svm = dlib.svm_c_trainer_linear()
    svm.set_c(10)
    classifier = svm.train(x, y)

    for sample, label in zip(x, y):
        print("%3i, %3i" % (label > 0, classifier(sample) > 0))

    # classifier models can also be pickled in the same was as any other python object.
    with open('saved_model.pickle', 'wb') as handle:
        pickle.dump(classifier, handle)

    sys.exit(app.exec_())

