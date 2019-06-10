
import pkgutil

import numina.drps
import numina.drps.drpbase
import numina.core.pipelineload as pload

from ..cli import main


drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')


def test_show_modes(capsys, monkeypatch):
    """Test that one instrument is shown"""

    def mockreturn():
        import numina.drps.drpbase
        import numina.core.pipelineload as pload
        drps = {}

        drp1 = pload.drp_load_data('numina', drpdata1)
        drps[drp1.name] = drp1
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ["Observing Mode: 'Fail' (fail)",
                " summary: Summary of fail recipe",
                " instrument: TEST1",
                "Observing Mode: 'Bias' (bias)",
                " summary: Summary of Bias recipe",
                " instrument: TEST1",
                "Observing Mode: 'Dark' (dark)",
                " summary: Summary of Dark recipe",
                " instrument: TEST1",
                "Observing Mode: 'Image' (image)",
                " summary: Summary of Image recipe",
                " instrument: TEST1",
                ""
                ]

    main(['show-modes'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_2_instruments(capsys, monkeypatch):
    """Test that two instruments are shown"""

    def mockreturn():
        drps = {}

        drp1 = pload.drp_load_data('numina', drpdata1)
        drp2 = pload.drp_load_data('numina', drpdata2)
        drps[drp1.name] = drp1
        drps[drp2.name] = drp2
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ['',
                "Observing Mode: 'Fail' (fail)",
                ' summary: Summary of fail recipe',
                ' instrument: TEST1',
                "Observing Mode: 'Bias' (bias)",
                ' summary: Summary of Bias recipe',
                ' instrument: TEST1',
                "Observing Mode: 'Dark' (dark)",
                ' summary: Summary of Dark recipe',
                ' instrument: TEST1',
                "Observing Mode: 'Image' (image)",
                ' summary: Summary of Image recipe',
                ' instrument: TEST1',
                "Observing Mode: 'Success' (success)",
                ' instrument: TEST2',
                ' summary: Summary of success recipe',
                "Observing Mode: 'Dark' (dark)",
                ' summary: Dark recipe',
                ' instrument: TEST2',
                ]

    main(['show-modes'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_modes_no_instruments(capsys, monkeypatch):
    """Test that no instruments are shown"""

    def mockreturn():
        return numina.drps.drpbase.DrpGeneric()

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ""

    main(['show-modes'])

    out, err = capsys.readouterr()

    assert out == expected


def test_show_modes_2_instruments_select_no(capsys, monkeypatch):
    """Test that two instruments are shown"""

    def mockreturn():
        drps = {}
        drp1 = pload.drp_load_data('numina', drpdata1)
        drp2 = pload.drp_load_data('numina', drpdata2)
        drps[drp1.name] = drp1
        drps[drp2.name] = drp2
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ["No instrument named: TEST3",
                ""
                ]

    main(['show-modes', '-i', 'TEST3'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_modes_no_instruments_select_no(capsys, monkeypatch):
    """Test that two instruments are shown"""

    def mockreturn():
        drps = {}
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ["No instrument named: TEST3",
                ""
                ]

    main(['show-modes', '-i', 'TEST3'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected
