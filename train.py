
import numpy as np

from optim.sgd import SGDOptimizer
from model.perceptron import PerceptronModel
from dataset.data import TwoClassesLinearDataset

from utils.loss import get_loss
from utils.toolbox import wb2line, LOGGER
from utils.drawtoolbox import draw_raw_data, init_draw, draw_cls_data, draw_loss, draw_final_cls

from config.config import config
from config.modeliniterA import modelA
from config.modeliniterB import modelB


def train(config):

    # init the dataset
    LOGGER.info('Init Dataset.')
    dataset = TwoClassesLinearDataset(config.data_num, config.dim, config.save_img_path)
    dataset.make_data()
    dataset.add_data()
    print('Dataset Info:')
    dataset.info()
    train_data = dataset.data_X
    train_label = dataset.data_y
    LOGGER.info('Dataset Img Path: {}'.format(config.save_img_path))
    draw_raw_data(dataset, config)

    # model A
    LOGGER.info('Start Model A')
    model_A = PerceptronModel()
    config_A = modelA()
    train_model(config_A, model_A, train_data, train_label, dataset)

    # model B
    LOGGER.info('Start Model B')
    model_B = PerceptronModel()
    config_B = modelB()
    train_model(config_B, model_B, train_data, train_label, dataset)


def train_model(config, model, train_data, train_label, dataset):

    # init model, loss and optim
    LOGGER.info('Model Info')
    model.model_init_info()
    loss_list = np.array([])
    sgd_optim = SGDOptimizer(model, config.learning_rate)
    LOGGER.info('Optimizer Info')
    sgd_optim.optim_info()
    draw_x = np.linspace(-5, 15, 10000)
    figure_board = init_draw(dataset)

    LOGGER.info('Start Training')
    print('max epoch set: {}'.format(config.max_epoch))

    # start train
    for epoch in range(config.max_epoch):

        # forward pass
        y_pred = model.forward_pass(train_data)

        # compute loss
        error_data = get_loss(train_data, y_pred, train_label, model)

        if error_data:
            [error_sample, error_label, loss] = error_data
            loss_list = np.append(loss_list, loss)
            # optim
            sgd_optim.sgd_update(error_sample, error_label)
            # draw
            draw_cls_data(draw_x, figure_board, False, model, wb2line, config)
        else:
            loss_list = np.append(loss_list, 0)
            LOGGER.warning('Train Complete')
            print('Iter Num: {}'.format(epoch))
            break
    
    draw_cls_data(draw_x, figure_board, True, model, wb2line, config)
    draw_final_cls(draw_x, model, wb2line, config, dataset)

    loss_x = np.linspace(0, loss_list.shape[0]-1, loss_list.shape[0])
    draw_loss(loss_x, loss_list, config)

    LOGGER.info('Draw Img of Loss, Classifiers and Optimization')
    print('Loss Img path: {}'.format(config.loss_path))
    print('Classifiers Img path: {}'.format(config.final_path))
    print('Optimization Img path: {}'.format(config.result_path))

    LOGGER.info('Model:')

    model.model_info()



if __name__ == '__main__':

    LOGGER.info('Program Start.')
    train_config = config()
    train(train_config)
    LOGGER.info('Program Done.')