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
import glob, os

def loadDataset(path) -> (dlib.vectors, dlib.vector):
    d = {}
    x = dlib.vectors()
    y = dlib.array()
    for f in glob.glob(path, recursive=True):
        (name, ext) = os.path.splitext(os.path.basename(f))
        chunks = name.split("_")
        if len(chunks) == 1:
            continue
        key = chunks[0]
        type = chunks[1]
        with open(f, 'r') as myfile:
            if key not in d:
                d[key] = {}
            raw = list(filter(lambda x: x != '', myfile.read().replace('\n', ' ').split(' ')))
            d[key][type] = list(map(lambda x: int(x), raw))

    print("Read " + str(len(d)) + " training sets")

    for item in d:
        xk = []
        xk.extend(d[item]['nose'])
        xk.extend(d[item]['eye1'])
        xk.extend(d[item]['eye2'])
        y.append(1 if d[item]['y'][0] == 0 else -1)
        if len(xk) != 402:
            print("ERROR: length of every xk should be 402")
        x.append(dlib.vector(xk))
    return x, y


(x, y) = loadDataset('h:/docs/progs/python/Hack112017/data/**/*.*')


# Make a training dataset.  Here we have just two training examples.  Normally
# you would use a much larger training dataset, but for the purpose of example
# this is plenty.  For binary classification, the y labels should all be either +1 or -1.
# x.append(dlib.vector([1, 2, 3, -1, -2, -3]))
# y.append(+1)
#
# x.append(dlib.vector([-1, -2, -3, 1, 2, 3]))
# y.append(-1)


# Now make a training object.  This object is responsible for turning a
# training dataset into a prediction model.  This one here is a SVM trainer
# that uses a linear kernel.  If you wanted to use a RBF kernel or histogram
# intersection kernel you could change it to one of these lines:
#  svm = dlib.svm_c_trainer_histogram_intersection()
svm = dlib.svm_c_trainer_radial_basis()
#svm = dlib.svm_c_trainer_linear()
#svm.be_verbose()
svm.set_c(10)

# Now train the model.  The return value is the trained model capable of making predictions.
classifier = svm.train(x, y)



# classifier models can also be pickled in the same was as any other python object.
with open('saved_model.pickle', 'wb') as handle:
    pickle.dump(classifier, handle)

