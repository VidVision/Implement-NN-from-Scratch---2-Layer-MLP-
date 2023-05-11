# References:
# https://heartbeat.fritz.ai/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432
# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        #                                                                           #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        W1 = self.weights['W1']           # (features x hidden size):(784xH)
        b1 = self.weights['b1']
        Z1 = X @ W1 + b1                  # (Nx784) @ (784xH) = (NxH)  - N: batch size
        A1 = self.sigmoid(Z1)             # (NxH)
        W2 = self.weights['W2']           # (HxK)
        b2 = self.weights['b2']
        Z2 = A1 @ W2 + b2                 # (NxK)
        p = self.softmax(Z2)              # Z2: (NxK)
        loss = self.cross_entropy_loss(p, y)
        accuracy = self.compute_accuracy(p, y)

        if mode != 'train':
            return loss, accuracy
        #############################################################################
        #                                                                           #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        # Calculate self.gradients['W1']
        # m = y.shape[0]
        m=len(y)
        grad = p  # softmax(Z2)
        grad[range(m), y] -= 1
        dL_dZ2 = grad / m           # (NxK)

        dZ2_dA1 = np.transpose(W2)  # (KxH)
        dL_dA1 = dL_dZ2@dZ2_dA1     # (NxH)

        dA1_dZ1 = self.sigmoid_dev(Z1)  # (NxH)
        dL_dZ1 = dL_dA1*dA1_dZ1         # (NxH)

        dZ1_dW1 = X                     # (Nx784)
        dL_dW1 = np.transpose(dZ1_dW1)@dL_dZ1  # (784xH)
        #------------------------------
        # Calculate self.gradients['W2']
        dZ2_dW2 = A1                           # (NxH)
        dL_dW2 = np.transpose(dZ2_dW2)@dL_dZ2  # (HxN)x(NxK)= (HxK)
        #-----------------------------
        # Calculate self.gradients['b1']
        dZ1_db1 = np.ones((np.shape(X)[0],))   #(N,)
        dL_db1 = np.transpose(dL_dZ1)@dZ1_db1  #(H,)
        #-----------------------------
        # Calculate self.gradients['b2']
        dZ2_db2 = np.ones((np.shape(X)[0],))   #(N,)
        dL_db2 = np.transpose(dL_dZ2)@dZ2_db2   #(N,)

        # Store the gradients in self.gradients
        self.gradients['W1'] = dL_dW1
        self.gradients['W2'] = dL_dW2
        self.gradients['b1'] = dL_db1
        self.gradients['b2'] = dL_db2

        return loss, accuracy


