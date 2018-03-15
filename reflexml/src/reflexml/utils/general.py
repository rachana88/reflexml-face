#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
file: general.py
description: general utilities for reflexml
author: Luke de Oliveira (lukedeo@vizzario.com)
"""

from errno import EEXIST
from hashlib import md5
import multiprocessing
from multiprocessing import Process, current_process
from os import makedirs
import random
from time import time as epoch_seconds


def get_dyn_sha():
    """
    issues a continually evolving SHA
    """
    m = md5()
    m.update('{pid}{secs}'.format(
        pid=current_process().pid,
        secs=epoch_seconds()
    ))
    return m.hexdigest()


def grouper(n, iterable):
    """
    yield n-sized chunks from an iterable (can be lazy)
    """
    if n == 1:
        for it in iterable:
            yield it
    else:
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk


def window_sample(seq, window_size, nb_samples):
    return [
        seq[i:(i + window_size)]
        for i in random.sample(xrange(len(seq) - window_size), nb_samples)
    ]


def window_select(seq, window_size):
    return [
        seq[(i - window_size):(i + window_size + 1)]
        for i in xrange(window_size, len(seq) - window_size - 1)
    ]


def safe_mkdir(path):
    '''
    Safe mkdir (i.e., don't create if already exists, 
    and no violation of race cond.)
    '''
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception


def async_call(functions, args=[], kwargs=[]):
    jobs = []
    pipe_list = []
    for i, f in enumerate(functions):

        recv_end, send_end = multiprocessing.Pipe(False)

        kw = {'hook': send_end}
        try:
            ar = args[i]
        except IndexError:
            ar = ()

        try:
            kw = kwargs[i]
        except IndexError:
            pass

        p = multiprocessing.Process(target=f, args=ar, kwargs=kw)
        jobs.append(p)
        pipe_list.append(recv_end)
        p.start()

    for proc in jobs:
        proc.join()

    result_list = [x.recv() for x in pipe_list]
    print result_list


def sha(s):
    """
    get a potentially value-inconsistent SHA of a python object.
    """
    m = md5()
    m.update(s.__repr__())
    return m.hexdigest()


def timestamp():
    """ 
    Returns a string of the form YYYYMMDDHHMMSS, where 
    HH is in 24hr time for easy sorting
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")


# we want to enforce an implementation of a static method using the ABC
# pattern. This hack allows you to enforce registration of a non-abstract
# version of a function

class abstractstatic(staticmethod):

    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


# this hack allows us to enforce the ABC implementation of a classmethod

class abstractclassmethod(classmethod):

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


def enforce_return_type(fn):
    """ Wrapper to ensure that a classmethod has a consistent return type with 
    the cls argument.
    Args:
    -----
        fn: classmethod to wrap
    Returns:
    --------
        the output of fn(cls, *args, **kwargs)
    Raises:
    -------
        asserts that isinstance(fn(cls, *args, **kwargs), cls) is true.
    """

    from functools import wraps

    @wraps(fn)
    def _typesafe_ret_type(cls, *args, **kw):
        o = fn(cls, *args, **kw)
        if not isinstance(o, cls):
            raise TypeError("Return type doesn't match specified "
                            "{}, found {} instead".format(cls, type(o)))
        return o

    _typesafe_ret_type.__doc__ = fn.__doc__
    return _typesafe_ret_type


def is_enforced_func(f):
    """
    Boolean test for whether or not a class method has been wrapped 
    with @enforce_return_type
    """
    return f.__code__.co_name == '_typesafe_ret_type'
