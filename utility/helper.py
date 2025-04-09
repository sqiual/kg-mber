import os
import logging
import datetime
from utility.parser import parse_args

args = parse_args()

from config.configurator import configs

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

class Logger(object):
    def __init__(self, log_configs=True):
        log_dir_path = './log/'
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        dataset_name = args.dataset        
        log_file = logging.FileHandler('{}/{}_{}.log'.format(log_dir_path, dataset_name, get_local_time()), 'a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        if log_configs:
            self.log(configs)

    def log(self, message, save_to_log=True, print_to_console=False):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_loss(self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=False):
        message = '[Epoch {:3d}] '.format(epoch_idx)
        for loss_name in loss_log_dict:
            message += '{}: {:.4f} '.format(loss_name, loss_log_dict[loss_name])
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(self, test_idx, eval_result, k, save_to_log=True, print_to_console=False):
        message = '[Test %d]: '% (test_idx)
        for metric in eval_result:
            message += '[' 
            for i in range(len(k)):
                message += '{}@{}: {:.5f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_val(self, eval_result, k, save_to_log=True, print_to_console=True):
        message = ''
        for metric in eval_result:
            message += '[' 
            for i in range(len(k)):
                message += '{}@{}: {:.5f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    update_flag = False
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
        update_flag = True  # find a better model
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop, update_flag


def early_stopping2(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop