# -*- coding: utf-8 -*-
"""
Created on Thu 24 15:45 2022

utilities

@author: NemotoS
"""

import torch
import numpy as np
import random
import logging
import datetime
import sys
from tqdm import tqdm

def fix_seed(seed=None,fix_gpu=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def init_logger(module_name,outdir="",tag="",level_console="warning",level_file="info"):
    level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
    if len(tag) == 0:
        tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(
        level = level_dic[level_file],
        filename = f"{outdir}/log_{tag}.txt",
        format = '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt = '%Y%m%d-%H%M%S'
    )
    logger = logging.getLogger(module_name)
    sh = TqdmLoggingHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
    )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def to_logger(logger,name="",obj=None,skip_keys=set(),skip_hidden=True):
    logger.info(name)
    for k,v in vars(obj).items():
        if k not in skip_keys:
            if skip_hidden:
                if not k.startswith("_"):
                    logger.info("   {0}: {1}".format(k,v))
            else:
                logger.info("   {0}: {1}".format(k,v))


class TqdmLoggingHandler(logging.Handler):
    def __init__(self,level=logging.NOTSET):
        super().__init__(level)

    def emit(self,record):
        try:
            msg = self.format(record)
            tqdm.write(msg,file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)