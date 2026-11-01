"""numina run with --validate and products whose value has no tags (#280)

The tags of a lines catalog are those of the directory of the calibration
tree where the DAL finds it.
"""

import logging

import pytest

from numina.core import BaseRecipe, Requirement
from numina.core.recipeinout import RecipeInput, TagsChecker
from numina.core.requirements import ObservationResultRequirement
from numina.exceptions import ValidationError
from numina.keydef import QueryAttribute
from numina.types.linescatalog import LinesCatalog

from tests.support.drpdef import make_drp

from .testdrp import SEEN, create_basedir, make_config, run_obs

PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"


class TaggedCatalog(LinesCatalog):
    """A lines catalog with tags, its value (an array) has no tags"""

    __tags__ = {"vph": QueryAttribute("vph", str)}


class CheckedInput(RecipeInput):
    __checkers__ = [TagsChecker()]


class CatalogRecipe(BaseRecipe):
    RecipeInput = CheckedInput
    obresult = ObservationResultRequirement()
    lines_catalog = Requirement(TaggedCatalog, "Catalog of lines")

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        SEEN.append(recipe_input.lines_catalog.tolist())
        return self.create_result()


CATALOG = {"id": 1, "mode": "catalog", "instrument": "TEST1", "images": ["image1.fits"], "labels": {"vph": "HR-R"}}


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    return create_basedir(drpmocker, tmp_path, monkeypatch, drp=make_drp({"catalog": CatalogRecipe}))


def write_catalog(basedir, *tags):
    directory = basedir.joinpath("calibs", "TEST1", PROFILE, "TaggedCatalog", *tags)
    directory.mkdir(parents=True)
    (directory / "lines.txt").write_text("6000.0 1.0\n")


def run(basedir, obs, *options):
    run_obs(make_config(basedir, {"calibsdir": str(basedir / "calibs")}), obs, "--validate", *options)


def test_tags_of_the_directory(basedir, caplog):
    write_catalog(basedir, "HR-R")
    with caplog.at_level(logging.WARNING):
        run(basedir, CATALOG)
    assert SEEN == [[6000.0, 1.0]]
    assert "are unknown" not in caplog.text


def test_tags_of_a_less_specific_directory(basedir, caplog):
    """A file without tags in the directory of the type: its tags are unknown, it is not checked"""
    write_catalog(basedir)
    with caplog.at_level(logging.WARNING):
        run(basedir, CATALOG)
    assert SEEN == [[6000.0, 1.0]]
    assert "requirement lines_catalog: the tags of the value of lines_catalog are unknown" in caplog.text


def test_value_in_the_observing_block(basedir, caplog):
    """A value given in the observing block has unknown tags"""
    (basedir / "data" / "lines.txt").write_text("6000.0 1.0\n")
    obs = dict(CATALOG, requirements={"lines_catalog": "lines.txt"})
    with caplog.at_level(logging.WARNING):
        run(basedir, obs)
    assert SEEN == [[6000.0, 1.0]]
    assert "the tags of the value of lines_catalog are unknown" in caplog.text


def test_tags_of_another_directory(basedir):
    """The catalog of another VPH is not selected"""
    write_catalog(basedir, "LR-U")
    with pytest.raises(ValueError, match="Required 'lines_catalog'"):
        run(basedir, CATALOG)
    assert SEEN == []


def test_tags_found_do_not_match(basedir, monkeypatch):
    """The tags with which the value was found are checked"""
    write_catalog(basedir, "HR-R")
    build = CatalogRecipe.build_recipe_input

    def build_wrong_tags(self, ob, dal):
        recipe_input = build(self, ob, dal)
        recipe_input.found_tags()["lines_catalog"] = {"vph": "LR-U"}
        return recipe_input

    monkeypatch.setattr(CatalogRecipe, "build_recipe_input", build_wrong_tags)
    with pytest.raises(ValidationError, match="the value of lines_catalog has vph=LR-U"):
        run(basedir, CATALOG)
