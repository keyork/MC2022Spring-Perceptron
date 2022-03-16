
import matplotlib.pyplot as plt


def draw_raw_data(dataset, config):
    '''
        draw the figure of data sample points and save
    '''

    x1 = dataset.X_cls_1[0, :].T
    y1 = dataset.X_cls_1[1, :].T
    x2 = dataset.X_cls_2[0, :].T
    y2 = dataset.X_cls_2[1, :].T

    fig, ax = plt.subplots()
    plt.xlim((-5, 15))
    plt.ylim((-5, 15))
    ax.plot(x1, y1, "bo", markersize=2, label='class1')
    ax.plot(x2, y2, "ro", markersize=2, label='class2')
    plt.savefig(config.save_img_path)
    plt.close()


def init_draw(dataset):
    '''
        init the figure of the process of optimizing
    '''

    x1 = dataset.X_cls_1[0, :].T
    y1 = dataset.X_cls_1[1, :].T
    x2 = dataset.X_cls_2[0, :].T
    y2 = dataset.X_cls_2[1, :].T

    fig, ax = plt.subplots()
    plt.xlim((-5, 15))
    plt.ylim((-5, 15))
    ax.plot(x1, y1, "bo", markersize=2, label='class1')
    ax.plot(x2, y2, "ro", markersize=2, label='class2')

    return ax


def draw_cls_data(x, ax, is_end, model, line, config):
    '''
        draw the figure of the process of optimizing
    '''

    if is_end:
        ax.plot(x, line(model.w, model.b, x), linestyle="-", color="red", label='sorting_line')
        plt.savefig(config.result_path)
        plt.close()
    else:
        ax.plot(x, line(model.w, model.b, x), linestyle="dashed", color="black", label='sorting_line')


def draw_final_cls(x, model, line, config, dataset):
    '''
        draw the final result of model and data
    '''

    x1 = dataset.X_cls_1[0, :].T
    y1 = dataset.X_cls_1[1, :].T
    x2 = dataset.X_cls_2[0, :].T
    y2 = dataset.X_cls_2[1, :].T

    fig, ax = plt.subplots()
    plt.xlim((-5, 15))
    plt.ylim((-5, 15))
    ax.plot(x1, y1, "bo", markersize=2, label='class1')
    ax.plot(x2, y2, "ro", markersize=2, label='class2')

    ax.plot(x, line(model.w, model.b, x), linestyle="-", color="red", label='sorting_line')
    plt.savefig(config.final_path)
    
    plt.close()


def draw_loss(loss_x, loss_list, config):
    '''
        draw the figure of loss value
    '''

    plt.plot(loss_x, loss_list, c='red')
    plt.savefig(config.loss_path)
    plt.close()