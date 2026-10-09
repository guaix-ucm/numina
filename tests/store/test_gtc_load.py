"""Load results in the format of GTC"""

import os

from numina.store.gtc.load import process_node
from numina.types.dataframe import DataFrame


def test_frame_node(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    node = {"name": "reduced_image", "typeid": 45, "value": {"path": "reduced.fits"}}

    name, obj = process_node(node)

    assert name == "reduced_image"
    assert isinstance(obj, DataFrame)
    assert obj.frame is None
    assert obj.filename == os.path.join(str(tmp_path), "reduced.fits")
