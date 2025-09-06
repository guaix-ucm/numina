from ..assembly import process_setup
from ..elements import SetupBlock


def test_setup1():
    setup_block = [{"values": {"ypos": [1, 2, 3, 4, 5, 6, 7]}}]
    res = process_setup("comp_store", "test", setup_block, "date")
    assert isinstance(res, SetupBlock)
