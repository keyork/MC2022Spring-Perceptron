
import numpy as np


class TwoClassesLinearDataset():
    '''
        class of dataset
        two class, each class contain 50 points (dim = 2)
        Args:
            - data_num: number of sample points
            - dim: dimensionality of sample points
            - save_img_path: the path save the figure of sample points
    '''

    def __init__(self, data_num, dim, save_img_path):
        
        self.data_num = data_num
        self.dim = dim
        self.save_img_path = save_img_path

        self.data_X = None
        self.data_y = None
        
        self.X_cls_1 = None
        self.y_cls_1 = None
        self.X_cls_2 = None
        self.y_cls_2 = None
    
    
    def make_data(self):

        self.X_cls_1 = np.random.randn(self.dim, self.data_num)
        self.y_cls_1 = np.ones((1, self.data_num))
        self.X_cls_1 = np.add(self.X_cls_1, [[np.random.randint(1,4)],[np.random.randint(8,10)]])

        self.X_cls_2 = np.random.randn(self.dim, self.data_num)
        self.y_cls_2 = -np.ones((1, self.data_num))
        self.X_cls_2 = np.add(self.X_cls_2, [[np.random.randint(7,10)],[np.random.randint(0,5)]])
        
        
    def add_data(self):
        
        self.data_X = np.concatenate((self.X_cls_1, self.X_cls_2), 1)
        self.data_y = np.concatenate((self.y_cls_1, self.y_cls_2), 1)


    def info(self):

        print('Data Shape: {}'.format(self.data_X.shape))
        print('Label Shape: {}'.format(self.data_y.shape))
        print('Class Num: {}'.format(2))
