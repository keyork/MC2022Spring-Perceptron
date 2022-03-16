
import logging
import colorlog


def log_color():

    color_logger = logging.getLogger('ROOT')
    color_logger.setLevel(logging.DEBUG)

    log_handler = logging.StreamHandler()
    log_handler.setLevel(logging.DEBUG)

    fmt_str = '%(log_color)s[%(levelname)s]: %(message)s (%(asctime)s)'

    log_colors = {
        'DEBUG': 'black, bg_white',
        'INFO': 'bg_green',
        'WARNING': 'bg_yellow',
        'ERROR': 'bg_red',
        'CRITICAL': 'bg_purple'
    }

    the_format = colorlog.ColoredFormatter(fmt_str, log_colors = log_colors)
    log_handler.setFormatter(the_format)
    color_logger.addHandler(log_handler)

    return color_logger

LOGGER = log_color()

def wb2line(w, b, x):
    '''
        convert (w, b) to y
    '''

    y = (1 / w[1][0]) * (- w[0][0] * x - b[0])
    return y