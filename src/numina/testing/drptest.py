import warnings
import pkgutil
import numina.drps.drpbase
import numina.core.pipelineload as pload

warnings.warn(
    "numina.testing.drptest is deprecated and will be removed in the next version of numina, "
    "it is only used by the tests of numina",
    DeprecationWarning,
    stacklevel=2,
)


def create_drp_test(names):
    """Function for creating DRP objects"""
    drps = {}
    for name in names:
        drpdata = pkgutil.get_data("numina.testing.drps", name)

        drp = pload.drp_load_data("numina", drpdata)
        drps[drp.name] = drp

    return numina.drps.drpbase.DrpGeneric(drps)
