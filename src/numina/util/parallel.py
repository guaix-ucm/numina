#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Pools of processes that log through the loggers of the main process"""

import contextlib
import logging
import logging.handlers
import multiprocessing


class _Dispatcher:
    """Handle the records of the processes with the loggers of the main process"""

    def handle(self, record):
        """Handle the record with the logger of its name, if the logger is enabled for its level."""
        logger = logging.getLogger(record.name)
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)


def _logger_levels():
    """The levels set in the loggers of this process, by name"""
    levels = {"": logging.getLogger().level}
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and logger.level != logging.NOTSET:
            levels[name] = logger.level
    return levels


def _init_process(queue, levels, initializer, initargs):
    """Send the records of the process to the queue, with the levels of the main process"""
    # The handlers inherited with the start method fork are removed,
    # the records reach the handlers of the main process through the queue
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.handlers.clear()
            logger.propagate = True
    root = logging.getLogger()
    root.handlers[:] = [logging.handlers.QueueHandler(queue)]
    # The root has the most verbose level of the main process, so that no
    # record shown there is discarded here; the main process filters the
    # records with the levels of its loggers. The levels are set only in
    # the loggers that exist: creating a logger before its package does
    # (as astropy, with its own class of logger) breaks the package
    root.setLevel(min(levels.values()))
    existing = logging.Logger.manager.loggerDict
    for name, level in levels.items():
        if name and isinstance(existing.get(name), logging.Logger):
            logging.getLogger(name).setLevel(level)
    if initializer is not None:
        initializer(*initargs)


@contextlib.contextmanager
def logging_pool(processes=None, initializer=None, initargs=(), context=None):
    """A multiprocessing.Pool whose processes log through the loggers of this process

    With the start methods spawn and forkserver (the default in Linux
    since Python 3.14), the processes of a pool do not have the logging
    configuration of the main process, and their messages are lost, for
    example, those of a recipe in numina run do not reach processing.log.
    In this pool, the messages of the processes are sent to the main
    process, that handles them with its loggers, as its own messages. The
    processes have the levels of the loggers of the main process when the
    pool is created.

    Parameters
    ----------
    processes : int, optional
        Number of processes, as in multiprocessing.Pool.
    initializer : callable, optional
        Called with initargs at the start of each process, after
        configuring its logging.
    initargs : tuple, optional
        Arguments of initializer.
    context : multiprocessing context, optional
        Context of the pool, the default context if None.

    Examples
    --------
    >>> with logging_pool(4) as pool:  # doctest: +SKIP
    ...     results = pool.map(function, values)
    """
    context = context or multiprocessing.get_context()
    queue = context.Queue()
    listener = logging.handlers.QueueListener(queue, _Dispatcher())
    try:
        with context.Pool(
            processes, initializer=_init_process, initargs=(queue, _logger_levels(), initializer, initargs)
        ) as pool:
            # the thread of the listener starts after creating the processes,
            # forking a process with threads may lead to deadlocks in the child
            # (a DeprecationWarning since Python 3.15); the records sent before
            # wait in the queue
            listener.start()
            try:
                yield pool
                # the records sent by the processes before closing the pool
                pool.close()
                pool.join()
            finally:
                listener.stop()
    finally:
        # stopping the listener puts a record in the queue, starting the
        # thread of the queue in this process; it is joined so that no
        # thread is left after the pool, a later fork would warn
        queue.close()
        queue.join_thread()
