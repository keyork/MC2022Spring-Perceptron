
import numpy as np

class SGDOptimizer():
    '''
        SGD optimizer
        use one of the error sample points to update W and b
        Args:
            - model: the model to be optimzed
            - learning_rate: learning rate
    '''

    def __init__(self, model, learning_rate):
        
        self.model = model
        self.lr = learning_rate
        
    
    def sgd_update(self, error_sample, error_label):
        
        self.model.w += self.lr * np.transpose([error_sample]) * error_label[0]
        self.model.b += self.lr * error_label[0]


    def optim_info(self):
        
        print('Optimizer: SGD')
        print('Learning Rate: {}'.format(self.lr))