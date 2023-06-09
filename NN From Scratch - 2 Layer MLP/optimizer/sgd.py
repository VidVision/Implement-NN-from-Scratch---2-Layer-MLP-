from ._base_optimizer import _BaseOptimizer
import numpy as np
class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)
        #############################################################################
        #                                                                           #
        #    1) Update model weights based on the learning rate and gradients       #
        #############################################################################
        W_list = []
        for key in model.weights.keys():
            W_list.append(key)
        if 'W1' in W_list:
            model.weights['W1'] -= self.learning_rate * model.gradients['W1']
        if 'W2' in W_list:
            model.weights['W2'] -= self.learning_rate * model.gradients['W2']
        if 'b1' in W_list:
            model.weights['b1'] -= self.learning_rate * model.gradients['b1']
        if 'b2' in W_list:
            model.weights['b2'] -= self.learning_rate * model.gradients['b2']
        return None
