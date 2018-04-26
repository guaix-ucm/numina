import pkgutil
import numina.drps.drpbase
import numina.core.pipelineload as pload


def create_drp_test(names):
    """Function for creating DRP objects"""
    drps = {}
    for name in names:
        drpdata = pkgutil.get_data('numina.drps.tests', name)

        drp = pload.drp_load_data('numina', drpdata)
        drps[drp.name] = drp

    return numina.drps.drpbase.DrpGeneric(drps)