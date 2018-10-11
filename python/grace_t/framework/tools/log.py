# -*- coding: utf8 -*-

import os
import logging
import logging.handlers

class SingleLevelFilter(logging.Filter):
    '''
    filter specific log level
    '''
    def __init__(self, passlevel, reject):
        '''
        init
        '''
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        '''
        real filter func
        '''
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            if self.passlevel >= logging.WARNING: 
                return (record.levelno >= self.passlevel)
            if self.passlevel == logging.DEBUG:
                if record.levelno >= logging.WARNING:
                    return False
                else:
                    return True
            else:
                return (record.levelno == self.passlevel)

#def init_log(file, level=logging.DEBUG, when="D", backup=7,
def init_log(file, level=logging.INFO, when="D", backup=7,
             format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
             datefmt="%m-%d %H:%M:%S"):
    '''
    init_log - initialize log module

    Args:
        + log_path: Log file path prefix.
            Log data will go to two files: log_path.log and log_path.log.wf
            Any non-exist parent directories will be created automatically
        + level: msg above the level will be displayed
            DEBUG < INFO < WARNING < ERROR < CRITICAL
            the default value is logging.INFO
        + when: how to split the log file by time interval
            + 'S' : Seconds
            + 'M' : Minutes
            + 'H' : Hours
            + 'D' : Days
            + 'W' : Week day
            default value: 'D'
        + format: format of the log
            default format:
            %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
            INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
        + backup: how many backup file to keep
            default value: 7

    Raises:
        + OSError: fail to create log directories
        + IOError: fail to open log file
    '''
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    for hd in logger.handlers:
        print hd
        logger.removeHandler(hd)
#    logger = logging.getLogger()
#    for hd in logger.handlers:
#        print hd
#        logger.removeHandler(hd)

    logger.setLevel(level)

    dir = os.path.dirname(file)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    handler = logging.handlers.TimedRotatingFileHandler(file + ".log",
                                                        when=when,
                                                        encoding="utf8",
                                                        backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    f1 = SingleLevelFilter(level, False)
    handler.addFilter(f1)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(file + ".log.wf",
                                                        when=when,
                                                        backupCount=backup)
    handler.setLevel(logging.WARNING)
    f2 = SingleLevelFilter(logging.WARNING, False)
    handler.removeFilter(f1)
    handler.addFilter(f2)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

'''
/* vim: set ts=4 sw=4 sts=4 tw=100 */
'''
