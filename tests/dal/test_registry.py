"""JSON store of the registry of reductions"""

import json
import os

import pytest

from numina.dal.registry import JSONStore, REGISTRY_FORMAT


@pytest.fixture
def store(tmp_path):
    return JSONStore(str(tmp_path / "db.json"))


def test_insert_get(store):
    docid = store.insert("tasks", {"state": 0})
    assert docid == 1
    assert store.get("tasks", 1) == {"id": 1, "state": 0}
    assert store.insert("tasks", {"state": 1}) == 2
    # ids are per collection
    assert store.insert("results", {}) == 1


def test_documents_are_copies(store):
    doc = {"tags": {"vph": "LR-U"}}
    store.insert("products", doc)
    doc["tags"]["vph"] = "changed"
    got = store.get("products", 1)
    assert got["tags"] == {"vph": "LR-U"}
    got["tags"]["vph"] = "changed"
    assert store.get("products", 1)["tags"] == {"vph": "LR-U"}


def test_update_find(store):
    store.insert("tasks", {"oblock_id": 1, "state": 0})
    store.insert("tasks", {"oblock_id": 2, "state": 0})
    store.insert("tasks", {"oblock_id": 1, "state": 0})
    store.update("tasks", 3, {"state": 2})

    assert [doc["id"] for doc in store.find("tasks", oblock_id=1)] == [1, 3]
    assert [doc["id"] for doc in store.find("tasks", oblock_id=1, state=2)] == [3]
    assert store.find("tasks", oblock_id=5) == []
    assert store.find("missing") == []


def test_missing_document(store):
    with pytest.raises(KeyError, match="document id=1 not found in collection 'tasks'"):
        store.get("tasks", 1)
    with pytest.raises(KeyError):
        store.update("tasks", 1, {})


def test_persistent(tmp_path):
    filename = str(tmp_path / "db.json")
    store = JSONStore(filename)
    assert not os.path.exists(filename)
    store.insert("tasks", {"state": 0})

    with open(filename) as fd:
        data = json.load(fd)
    assert data["format"] == REGISTRY_FORMAT
    assert data["version"] == 1

    other = JSONStore(filename)
    assert other.get("tasks", 1) == {"id": 1, "state": 0}
    # the next id continues
    assert other.insert("tasks", {}) == 2


def test_transaction_writes_once(store, monkeypatch):
    writes = []
    original = store._write

    def counting_write():
        writes.append(1)
        original()

    monkeypatch.setattr(store, "_write", counting_write)
    with store.transaction():
        store.insert("tasks", {})
        store.insert("tasks", {})
        assert writes == []
    assert writes == [1]


def test_transaction_rollback(store):
    store.insert("tasks", {"state": 0})
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.update("tasks", 1, {"state": 2})
            store.insert("tasks", {})
            raise RuntimeError("error")

    assert store.get("tasks", 1)["state"] == 0
    assert store.find("tasks") == [{"id": 1, "state": 0}]
    # the file was not changed either
    assert JSONStore(store.filename).find("tasks") == [{"id": 1, "state": 0}]


def test_failed_write_keeps_file(store, monkeypatch):
    import numina.dal.registry as registry

    store.insert("tasks", {"state": 0})
    with open(store.filename) as fd:
        before = fd.read()

    def failing_dump(obj, fp, **kwargs):
        fp.write("partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(registry.json, "dump", failing_dump)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.insert("tasks", {})

    with open(store.filename) as fd:
        assert fd.read() == before
    assert os.listdir(os.path.dirname(store.filename)) == ["db.json"]


@pytest.mark.parametrize(
    "content, msg",
    [
        ('{"version": 2, "database": {}}', "is not a numina registry"),
        ("[]", "is not a numina registry"),
        (f'{{"format": "{REGISTRY_FORMAT}", "version": 99, "collections": {{}}}}', "unsupported version 99"),
    ],
)
def test_invalid_file(tmp_path, content, msg):
    filename = tmp_path / "db.json"
    filename.write_text(content)
    with pytest.raises(ValueError, match=msg):
        JSONStore(str(filename))


def test_store_interface(store):
    from numina.dal.registry import Store

    assert isinstance(store, Store)

    class Incomplete(Store):
        def insert(self, collection, doc):
            return 1

    with pytest.raises(TypeError):
        Incomplete()


def test_registry_with_other_store(tmp_path):
    """The registry uses any Store"""
    from numina.dal.registry import Registry

    store = JSONStore(str(tmp_path / "other.json"))
    registry = Registry("unused.json", basedir=str(tmp_path), store=store)
    store.insert("products", {"instrument": "TEST1", "profile": "P", "qc": "GOOD", "type": "T"})
    assert registry.select_product("TEST1", "P")["id"] == 1
    assert not (tmp_path / "unused.json").exists()


def test_select_results(tmp_path):
    from numina.dal.registry import Registry

    registry = Registry(str(tmp_path / "db.json"), basedir=str(tmp_path))
    store = registry.store
    task = store.insert("tasks", {"request_params": {"instrument_configuration": "P1"}})
    common = {"instrument": "TEST1", "mode": "bias"}
    # recorded without profile, from the task
    store.insert("results", dict(common, qc="GOOD", task_id=task))
    store.insert("results", dict(common, qc="GOOD", profile="P1"))
    store.insert("results", dict(common, qc="BAD", profile="P1"))
    store.insert("results", dict(common, qc="GOOD", profile="P2"))
    store.insert("results", {"instrument": "TEST1", "mode": "dark", "qc": "GOOD", "profile": "P1"})

    results = registry.select_results("TEST1", "P1", "bias")
    assert [res["id"] for res in results] == [2, 1]
    assert registry.select_results("TEST1", "P3", "bias") == []
