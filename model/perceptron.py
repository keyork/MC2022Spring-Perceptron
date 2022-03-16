
import numpy as np

class PerceptronModel():
    '''
        perceptron model
        assuming input data's dim is 2
        Args:
            None
    '''

    def __init__(self):
        
        self.w = np.random.randn(2,1)
        self.b = np.random.randn(1)


    def forward_pass(self, input):
        '''
            input shape: (2, N)
        '''

        pred_result = np.dot(np.transpose(self.w), input) + self.b
        return pred_result

    
    def model_info(self):

        print('The expression is: f(x) = sign(w^T·x+b)')
        print('which:')
        print('w = [{},'.format(self.w[0,:][0]))
        print('     {}]'.format(self.w[1, :][0]))
        print('b = [{}]'.format(self.b[0]))


    def model_init_info(self):

        print('w = [{},'.format(self.w[0,:][0]))
        print('     {}]'.format(self.w[1, :][0]))
        print('b = [{}]'.format(self.b[0]))
