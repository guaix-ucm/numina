#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Progress bar that logs complete lines."""


class ProgressBarLines:
    """Progress bar that logs complete lines."""

    def __init__(self, total=None, logger=None):
        if total is None or logger is None:
            raise ValueError("Total and logger must be provided")
        self.total = total
        self.current = 0
        self.shown_milestones = set([0])
        self.logger = logger
        self.progress_line = "0%"
        self.logger.info(self.progress_line)

    def update(self, step=1):
        """Update progress."""
        self.current += step
        percent = (self.current / self.total) * 100

        for milestone in range(10, 101, 10):
            if percent >= milestone and milestone not in self.shown_milestones:
                self.shown_milestones.add(milestone)
                self.progress_line += f" {milestone}%"
                self.logger.info(self.progress_line)
