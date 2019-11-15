## RBFN.py          Dana Hughes                 14-Nov-2019
##
## A simple class for creating, training and performing inference using an 
## RBFN.
## 
## Author: Dana Hughes
## Contact: danathughes@gmail.com
## 
## Revisions:
##    14-Nov-2019    Initial version containing a simple RBFN class.
##

"""
"""


import numpy as np
import tensorflow as tf


class RBFN(object):

    def __init__(self, input_size, hidden_size, beta=1.0):
        """ 
        Create a Radial Basis Function Network using Gaussian Kernels

        Arguments:
          input_size  - the dimensionality of the input
          hidden_size - the number of kernels / hidden units to use
          beta        - the shape parameter of the Gaussian kernel
        """

        # The learned weights and centers
        self.learned_weights = np.zeros((hidden_size,))
        self.learned_centers = np.zeros((hidden_size, input_size))

        # Tensorflow model
        if tf.get_default_session() is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = tf.get_default_session()


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigma = sigma

        # Possible inputs -- states and actions, RBF centers, kernel weights
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.input_size))
        self.centers = tf.placeholder(tf.float32, shape=(self.hidden_size, self.input_size))

        self.weights = tf.placeholder(tf.float32, (self.hidden_size,))
        self.targets = tf.placeholder(tf.float32, (None,))


        # Build the relevant dataflow pipes to calculate the interpolation matrix, 
        # compute the weight matrix, and predict outputs

        # Outputs:
        # interpolation_matrix -- matrix of kernel functions between input data and centers
        # fit_weights -- fitted weights of each kernel
        # output -- predicted output

        self.interpolation_matrix = self._kernel(self.inputs, self.centers)
        self.fit_weights = self._weight_calculator(self.interpolation_matrix, self.targets)
        self.output = tf.matmul(self.interpolation_matrix, tf.reshape(self.weights, (self.hidden_size,1)))
        self.output = tf.reshape(self.output, (-1,))


    def _kernel(self, x, centers):
        """
        """

        tiled_x = tf.tile(tf.reshape(x, (-1,1,self.input_size)), [1, self.hidden_size,1])
        reshaped_centers = tf.reshape(self.centers, (1,)+tuple(self.centers.shape))
        squared_distances = tf.reduce_sum(tf.square(tiled_x - reshaped_centers), axis=2)

        return tf.exp(-self.sigma*squared_distances)


    def _pinv(self, a, rcond=1e-15):
        """
        """

        s,u,v = tf.svd(a)

        limit = rcond*tf.reduce_max(s)
        non_zero = tf.greater(s, limit)

        reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(tf.size(s)))
        lhs = tf.matmul(v, tf.matrix_diag(reciprocal))

        return tf.matmul(lhs, u, transpose_b=True)


    def _weight_calculator(self, kernel_matrix, targets):
        """
        """

        weights = tf.matmul(self._pinv(self.interpolation_matrix), tf.reshape(self.targets, (-1,1)))
        return tf.reshape(weights, (-1,))


    def set_centers(self, centers):
        """
        """

        # Make sure that the provided centers are the correct weight.
        # NOTE: Implement an assert...

        self.learned_centers = centers


    def fit(self, X, Y, centers=None):
        """
        """

        # Assert that the size of the centers are correct (if present), or that there are 
        # __at least__ the correct number of examples in X


        if centers == None:
            self.set_centers(X[np.random.choice(range(len(X)), self.hidden_size)])
        else:
            self.set_centers(centers)

        self.learned_weights = self.sess.run(self.fit_weights, feed_dict = { self.inputs: X,
                                                                             self.targets: Y,
                                                                             self.centers: self.learned_centers })


    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """

        return self.sess.run(self.output, feed_dict = { self.inputs: X, 
                                                         self.centers: self.learned_centers,
                                                         self.weights: self.learned_weights })



## TEST DATA
x, y = np.meshgrid(np.linspace(-5,5,20), np.linspace(-5,5,20))
z = (np.sin(np.sqrt((x-2.)**2 + (y-1.)**2)) - np.sin(np.sqrt((x+2.)**2 + (y+4.)**2)))/2.

X = np.asarray(list(zip(x.flatten(), y.flatten())))
Y = z.flatten()