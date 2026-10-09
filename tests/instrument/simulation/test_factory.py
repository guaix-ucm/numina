"""Run counters of the simulations"""

from numina.instrument.simulation.factory import PersistentRunCounter, RunCounter


def test_run_counter():
    counter = RunCounter("r%04d.fits", last=5)
    assert counter.runstring() == "r0005.fits"
    assert counter.runstring() == "r0006.fits"


def test_persistent_run_counter(tmp_path):
    pstore = tmp_path / "index.json"

    # the file is created if it does not exist
    with PersistentRunCounter("r%04d.fits", pstore=str(pstore)) as counter:
        assert pstore.exists()
        assert counter.runstring() == "r0001.fits"
        assert counter.runstring() == "r0002.fits"

    # the next run number is stored and read
    with PersistentRunCounter("r%04d.fits", pstore=str(pstore)) as counter:
        assert counter.runstring() == "r0003.fits"
