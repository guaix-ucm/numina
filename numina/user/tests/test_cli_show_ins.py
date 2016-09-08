
import pkgutil

import numina.drps

from ..cli import main


drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')


def test_show_instrument(capsys, monkeypatch):
    """Test that one instrument is shown"""

    def mockreturn():
        import numina.drps.drpbase
        import numina.core.pipelineload as pload
        drps = {}

        drp1 = pload.drp_load_data(drpdata1)
        drps[drp1.name] = drp1
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ("Instrument: TEST1\n"
                " has configuration 'default'\n"
                " has pipeline 'default', version 1\n"
                )

    main(['show-instruments'])

    out, err = capsys.readouterr()
    assert out == expected


def test_show_2_instruments(capsys, monkeypatch):
    """Test that two instruments are shown"""

    def mockreturn():
        import numina.drps.drpbase
        import numina.core.pipelineload as pload
        drps = {}

        drp1 = pload.drp_load_data(drpdata1)
        drp2 = pload.drp_load_data(drpdata2)
        drps[drp1.name] = drp1
        drps[drp2.name] = drp2
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ["Instrument: TEST2",
                " has configuration 'default'",
                " has pipeline 'default', version 1",
                "Instrument: TEST1",
                " has configuration 'default'",
                " has pipeline 'default', version 1",
                ""
                ]

    main(['show-instruments'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_no_instrument(capsys, monkeypatch):
    """Test that no instruments are shown"""

    def mockreturn():
        import numina.drps.drpbase
        return numina.drps.drpbase.DrpGeneric()

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ""

    main(['show-instruments'])

    out, err = capsys.readouterr()

    assert out == expected
