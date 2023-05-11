# Do not use packages that are not in standard distribution of python

# References:
# https://deepnotes.io/softmax-crossentropy
# https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu

import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        prob = None
        #############################################################################
        #                                                                            #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        # https://deepnotes.io/softmax-crossentropy
        # to make softmax func. numerically stable normalize the values in the vector,
        # by multiplying the numerator and denominator with a constant C
        # generally used log(C) is -max(a) to avoid overflowing.
        numer = np.exp(scores - np.max(scores, axis=1, keepdims=True)) # exp val at each row
        denum = np.sum(numer, axis=1,keepdims=True) #sum of all exp values(rows)
        prob = numer/denum
        return prob



        return prob

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        loss = None
        #############################################################################
        #                                                                           #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################
        # https://deepnotes.io/softmax-crossentropy
        # x_pred is probability so no need to use softmax here
        m = np.shape(x_pred)[0]
        # select the log_prediction in each row (range(m)) corresponding to the target of that row (y)
        ll = -np.log(x_pred[range(m), y])
        loss = np.sum(ll) / m
        return loss

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        acc = None
        #############################################################################
        #                                                                           #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        pred_class = np.argmax(x_pred, axis=1) #get the index (class) of max prob
        acc = np.count_nonzero(pred_class == y)/len(y)
        # cor_pred = np.where(pred_class == y)
        # acc = len(cor_pred)/len(y)
        return acc

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        '''
        out = None
        #############################################################################
        #        Comput the sigmoid activation on the input                          #
        #############################################################################
        out = 1.0 / (1.0 + np.exp(-X))

        return out

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        ds = None
        #############################################################################
        #                                                                           #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
        sig_x = 1.0 / (1.0 + np.exp(-x))
        ds = sig_x*(1-sig_x)

        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        '''
        out = None
        #############################################################################
        # Comput the ReLU activation on the input                                   #
        #############################################################################
        # https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
        out = np.maximum(0.0, X)
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        '''
        out = None
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        # The gradient of ReLU is 1 for x>0 and 0 for x<0
        # https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu
        out = (X > 0) * 1
        return out
