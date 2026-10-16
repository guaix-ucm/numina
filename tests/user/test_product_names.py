"""numina run with the template product_tmpl of the names of the products"""

import astropy.io.fits as fits
import pytest

from numina.dal.registry import Registry

from .testdrp import BIAS, DB_OPTIONS, IMAGE, create_basedir, make_config, run_obs


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    return create_basedir(drpmocker, tmp_path, monkeypatch)


def test_default_product_names(basedir):
    run_obs(make_config(basedir), BIAS)
    assert (basedir / "obsid1_results" / "master_bias.fits").is_file()


def test_unique_product_names(basedir, seen):
    """The products have the uuid in their name, and are found in the registry"""
    config = make_config(basedir, run_values={"product_tmpl": "{name}_{uuid}"})
    run_obs(config, BIAS, *DB_OPTIONS)

    registry = Registry(str(basedir / "numina-db.json"))
    (product,) = registry.products(type="MasterBias")
    assert product["content"].endswith(f"master_bias_{product['uuid']}.fits")
    assert fits.getheader(basedir / product["content"])["UUID"] == product["uuid"]

    # the product is used by other reductions
    run_obs(config, IMAGE, *DB_OPTIONS)
    assert seen == [product["uuid"]]


def test_two_runs_have_different_names(basedir):
    config = make_config(basedir, run_values={"product_tmpl": "{name}_{uuid}"})
    run_obs(config, BIAS, *DB_OPTIONS)
    run_obs(config, BIAS, *DB_OPTIONS)
    names = {product["content"].rsplit("/", 1)[-1] for product in Registry(str(basedir / "numina-db.json")).products()}
    assert len(names) == 2


def test_invalid_product_template(basedir):
    config = make_config(basedir, run_values={"product_tmpl": "{name}_{date}"})
    with pytest.raises(ValueError, match="field '{date}' not allowed in template"):
        run_obs(config, BIAS)
