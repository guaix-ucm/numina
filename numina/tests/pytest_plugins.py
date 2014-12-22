#
# Copyright 2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

import pytest


def pytest_addoption(parser):
    parser.addoption("--remote-data", action="store_true", default=False,
                     help="run tests with online data")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers",
        "remote: mark test to run with online data"
        )


def pytest_runtest_setup(item):
    if ('remote' in item.keywords and
            not item.config.getoption("--remote-data")):

        pytest.skip("need --remote-data option to run")
