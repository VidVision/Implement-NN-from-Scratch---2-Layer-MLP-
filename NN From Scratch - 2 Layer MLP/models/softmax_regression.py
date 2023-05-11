# References:
# https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (optional ReLU activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        #                                                                           #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        W = self.weights['W1']           # (features x classes):(784xK)
        Z = X @ W                        # (Nx784) @ (784xK) = (NxK)  - N: batch size
        A = self.ReLU(Z)                 # A: (NxK)
        p = self.softmax(A)              # A: (NxK)
        loss = self.cross_entropy_loss(p, y)
        accuracy = self.compute_accuracy(p, y)

        if mode != 'train':
            return loss, accuracy

        #############################################################################
        #                                                                           #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        # dL/dW = dL/dA x dA/dZ x dZ/dW
        # https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax
        #  We can simplify dL/dSoftmax * dSoftmax/dReLU using above source to pi-yi
        # dL_dA = y - p                       # (N x K)
        # m = y.shape[0]
        m = len(y)
        grad = p  # softmax(ReLU)
        grad[range(m), y] -= 1
        dL_dA = grad / m                   # (N x K)

        # following OH slides
        dA_dZ = self.ReLU_dev(Z)            # (N x K)
        dZ_dW = X                           # (N x 784)
        dL_dZ = dL_dA * dA_dZ               # (N x K)
        dL_dW = np.transpose(dZ_dW)@dL_dZ   # (784 x K)
        self.gradients['W1'] = dL_dW

        return loss, accuracy





        


