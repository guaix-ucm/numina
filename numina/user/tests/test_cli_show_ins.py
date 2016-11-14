
import pkgutil

import pytest

import numina.drps
import numina.drps.drpbase
import numina.core.pipelineload as pload

from ..cli import main


drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')

expecte0 = [""]


expecte1 = ["Instrument: TEST1",
            " has configuration 'default'",
            " has pipeline 'default', version 1",
            ""
            ]


expecte2 = ["Instrument: TEST2",
            " has configuration 'default'",
            " has pipeline 'default', version 1",
            "Instrument: TEST1",
            " has configuration 'default'",
            " has pipeline 'default', version 1",
            ""
            ]


def mockreturn0():
    return numina.drps.drpbase.DrpGeneric()


def mockreturn1():
    drps = {}
    drp1 = pload.drp_load_data('numina', drpdata1)
    drps[drp1.name] = drp1
    return numina.drps.drpbase.DrpGeneric(drps)


def mockreturn2():
    drps = {}
    drp1 = pload.drp_load_data('numina', drpdata1)
    drp2 = pload.drp_load_data('numina', drpdata2)
    drps[drp1.name] = drp1
    drps[drp2.name] = drp2
    return numina.drps.drpbase.DrpGeneric(drps)


@pytest.mark.parametrize("drpsfunc, expected",
                         [(mockreturn0, expecte0),
                          (mockreturn1, expecte1),
                          (mockreturn2, expecte2)
                          ]
                         )
def test_show_instrument(capsys, monkeypatch, drpsfunc, expected):
    """Test that no instruments are shown"""

    monkeypatch.setattr(numina.drps, "get_system_drps", drpsfunc)

    main(['show-instruments'])


    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


@pytest.mark.parametrize("drpsfunc", [mockreturn0, mockreturn1, mockreturn2])
def test_show_no_instruments_i(capsys, monkeypatch, drpsfunc):
    """Test that two instruments are shown"""

    monkeypatch.setattr(numina.drps, "get_system_drps", drpsfunc)

    expected = ["No instrument named: TEST3",
                ""
                ]

    main(['show-instruments', '-i', 'TEST3'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected
