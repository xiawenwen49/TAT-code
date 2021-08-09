import logging
import time
import os
import socket


def set_up_log(args, sys_argv):
    log_dir = args.log_dir
    dataset_log_dir = os.path.join(log_dir, args.dataset) #.rstrip('.txt')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(dataset_log_dir):
        os.mkdir(dataset_log_dir)
    file_path = os.path.join(dataset_log_dir, '{}.log'.format( time.strftime('%Y_%m_%d_%H_%M_%S') ))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.save_log: # add file handler
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info('Create log file at {}'.format(file_path))

    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('-'*40)
    logger.info('Full args parsed:')
    for key, value in vars(args).items():
        logger.info('{}={}'.format(key, value))
    logger.info('-'*40)

    return logger


def save_performance_result(args, logger, metrics):
    summary_file = args.summary_file
    if summary_file != 'test':
        summary_file = os.path.join(args.log_dir, summary_file)
    else:
        return
    dataset = args.dataset
    val_metric, no_val_metric = metrics
    model_name = '-'.join([args.model, args.feature, str(args.prop_depth)])
    seed = args.seed
    log_name = os.path.split(logger.handlers[1].baseFilename)[-1]
    server = socket.gethostname()
    line = '\t'.join([dataset, model_name, str(seed), str(round(val_metric, 4)), str(round(no_val_metric, 4)), log_name, server]) + '\n'
    with open(summary_file, 'a') as f:
        f.write(line)  # WARNING: process unsafe!





