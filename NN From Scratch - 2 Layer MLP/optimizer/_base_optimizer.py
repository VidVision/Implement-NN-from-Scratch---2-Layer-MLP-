import models


class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        '''
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        '''

        #############################################################################
        #                                                                           #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################
        # https://visualstudiomagazine.com/articles/2017/09/01/neural-network-l2.aspx
        # in oredr to prevent overfitting, we need to add regularization term
        # "Note that it’s standard practice to not apply the L2 penalty to the hidden node biases or the output node biases.
        # The reasoning is rather subtle, but briefly and informally: A single bias value with large magnitude
        # isn't likely to lead to model overfitting because the bias can be compensated for by multiple associated weights."
        W_list = []
        for key in model.weights.keys():
            W_list.append(key)
        # print(W_list)
        if 'W1' in W_list:
            model.gradients['W1'] += model.weights['W1'] * self.reg
        if 'W2' in W_list:
            model.gradients['W2'] += model.weights['W2'] * self.reg
        return None
