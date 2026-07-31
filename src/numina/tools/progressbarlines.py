#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Progress bar that logs complete lines."""

import time


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
        # show initial line with elapsed time and estimated time remaining
        self.logger.info(self.progress_line)
        self.start_time = time.time()

        # Fixed width: length of the longest possible line
        full_line = "0%" + "".join(f" {m}%" for m in range(10, 101, 10))
        self.label_width = len(full_line)

    def update(self, step=1):
        """Update progress."""
        self.current += step
        percent = (self.current / self.total) * 100

        for milestone in range(10, 101, 10):
            if percent >= milestone and milestone not in self.shown_milestones:
                self.shown_milestones.add(milestone)
                elapsed = time.time() - self.start_time
                eta = elapsed * (100 - milestone) / milestone

                self.progress_line += f" {milestone}%"
                padded = self.progress_line.ljust(self.label_width, "_")
                line_to_show = f"{padded} (elap={self._fmt(elapsed)}|left={self._fmt(eta)}|exp={self._fmt(elapsed + eta)})"
                self.logger.info(line_to_show)

    @staticmethod
    def _fmt(seconds):
        """Format seconds as H:MM:SS or M:SS, like tqdm."""
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"
