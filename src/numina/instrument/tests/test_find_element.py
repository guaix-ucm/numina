"""Selection of instrument configurations with find_instrument"""

import logging

import pytest

from ..assembly import find_instrument
from ..collection import load_paths_store

# Profiles in numina/drps/tests/configs, TEST1 has two
TEST1_PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"  # since 2016-06-01
TEST1_OLD_PROFILE = "6ad5dc90-6b15-43b7-abb5-b07340e19f41"  # 2014-06-01 to 2016-06-01
CLODIA_PROFILE = "44077557-32ab-43f2-8f2a-5ddb813c03df"


@pytest.fixture(scope="module")
def comp_store():
    return load_paths_store(["numina.drps.tests.configs"])


def uuid_of(element):
    return str(element["origin"].uuid)


def test_no_date_uses_most_recent(comp_store, caplog):
    with caplog.at_level(logging.WARNING, logger="numina.instrument.assembly"):
        element = find_instrument(comp_store, "TEST1", None)

    assert uuid_of(element) == TEST1_PROFILE
    assert f"using the most recent: uuid={TEST1_PROFILE}" in caplog.text


@pytest.mark.parametrize(
    "date, expected",
    [("2017-01-01T00:00:00", TEST1_PROFILE), ("2015-01-01T00:00:00", TEST1_OLD_PROFILE)],
)
def test_date_selects_without_warning(comp_store, caplog, date, expected):
    with caplog.at_level(logging.WARNING, logger="numina.instrument.assembly"):
        element = find_instrument(comp_store, "TEST1", date)

    assert uuid_of(element) == expected
    assert caplog.text == ""


@pytest.mark.parametrize(
    "keyval, by_key, expected",
    [("CLODIA", "name", CLODIA_PROFILE), (TEST1_OLD_PROFILE, "uuid", TEST1_OLD_PROFILE)],
)
def test_no_date_one_candidate_without_warning(comp_store, caplog, keyval, by_key, expected):
    """Only one configuration: by name with a single one, or by uuid"""
    with caplog.at_level(logging.WARNING, logger="numina.instrument.assembly"):
        element = find_instrument(comp_store, keyval, None, by_key=by_key)

    assert uuid_of(element) == expected
    assert caplog.text == ""


def test_not_found(comp_store):
    with pytest.raises(ValueError, match="Not found instrument name=TEST1 for date=2010-01-01"):
        find_instrument(comp_store, "TEST1", "2010-01-01")
