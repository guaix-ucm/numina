"""Selection of the pipeline in numina run"""

import pytest

from numina.user.baserun import run_reduce
from numina.user.helpers import create_datamanager

# The DRP TEST1, with two pipelines that use different recipes for the same mode
DRP_TWO_PIPELINES = """
name: TEST1
configurations:
  path: numina.drps.tests.configs
modes:
  - key: image
    name: Image
    summary: Image mode
    description: Image mode
pipelines:
  default:
    version: 1
    recipes:
      image: numina.core.utils.AlwaysSuccessRecipe
  alt:
    version: 1
    recipes:
      image: numina.core.utils.OBSuccessRecipe
"""

RECIPE_OF_PIPELINE = {"default": "AlwaysSuccessRecipe", "alt": "OBSuccessRecipe"}


@pytest.fixture
def datamanager(drpmocker, run_config, tmp_path):
    drpmocker.add_drp("TEST1", DRP_TWO_PIPELINES)
    datadir = tmp_path / "data"
    datadir.mkdir()
    run_config["tool.run"]["datadir"] = str(datadir)
    return create_datamanager(run_config, None)


def add_ob(datamanager, ob_pipeline):
    ob = {"id": 1, "mode": "image", "instrument": "TEST1", "images": []}
    if ob_pipeline is not None:
        ob["pipeline"] = ob_pipeline
    datamanager.backend.add_obs([ob])


@pytest.mark.parametrize(
    "requested, ob_pipeline, expected",
    [
        (None, None, "default"),  # nothing defined
        (None, "alt", "alt"),  # from the OB
        ("alt", None, "alt"),  # from the request (the -p option)
        ("alt", "default", "alt"),  # the request overrides the OB
        ("default", "alt", "default"),
    ],
)
def test_pipeline_selection(datamanager, requested, ob_pipeline, expected):
    add_ob(datamanager, ob_pipeline)

    task = run_reduce(datamanager, 1, pipeline=requested)

    assert task.state == 2
    # The pipeline used is recorded in the task
    assert task.request_params["pipeline"] == expected
    assert task.request_runinfo["pipeline"] == expected
    assert task.request_runinfo["recipe_class"] == RECIPE_OF_PIPELINE[expected]


def test_pipeline_not_found(datamanager):
    add_ob(datamanager, None)

    msg = r"pipeline 'other' not found in DRP 'TEST1', available pipelines are \['default', 'alt'\]"
    with pytest.raises(KeyError, match=msg):
        run_reduce(datamanager, 1, pipeline="other")
