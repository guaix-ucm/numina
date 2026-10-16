"""The requirements of the observation result reach the recipe"""

from tests.support.drpdef import make_drp
from .testdrp import run_obs

DRP_TEST1 = make_drp({"param": "tests.user.testdrp.ParamRecipe"})


def test_ob_requirements(drpmocker, run_config, tmp_path, seen):
    drpmocker.add_drp("TEST1", DRP_TEST1)
    datadir = tmp_path / "data"
    datadir.mkdir()
    run_config["tool.run"]["datadir"] = str(datadir)
    obs = [
        {"id": 1, "mode": "param", "instrument": "TEST1", "images": []},
        {"id": 2, "mode": "param", "instrument": "TEST1", "images": [], "requirements": {"value": 2}},
    ]
    run_obs(run_config, obs)

    assert seen == [1, 2]
