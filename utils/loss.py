
import numpy as np


def get_loss(data, pred_result, label, model):
    '''
        data: (2,100)
        lable: (1,100), 1/-1
        pred_result: (1,100), nums
    '''
    
    error_index = np.where(label[0]*pred_result[0] <= 0)
    error_data = data[:, error_index[0]]
    error_label = label[:, error_index[0]]
    error_pred = pred_result[:, error_index[0]]
    loss = - np.sum(error_pred * error_label) / np.linalg.norm(model.w, ord=2)

    if error_index[0].size:
        select_index = np.random.randint(0, error_data.shape[1])
        select_sample = error_data[:, select_index]
        return [select_sample, error_label[:, select_index], loss]
    else:
        return 0
