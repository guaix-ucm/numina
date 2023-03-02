
import pkgutil

import pytest

import numina
import numina.drps
import numina.drps.drpbase
import numina.core.pipelineload as pload

from ..cli import main


drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')

expecte0 = [""]


expecte1 = ["Instrument: TEST1",
            " version is '1'",
            " has configuration 'Old configuration' uuid=6ad5dc90-6b15-43b7-abb5-b07340e19f41",
            " has configuration 'Default configuration' uuid=225fcaf2-7f6f-49cc-972a-70fd0aee8e96",
            " has datamodel 'numina.datamodel.DataModel'",
            " has pipeline 'default', version 1",
            ""
            ]


expecte2 = ["Instrument: TEST2",
            f" version is '{numina.__version__}'",
            " has datamodel 'numina.datamodel.DataModel'",
            " has configuration 'Default configuration' uuid=9c21b315-9231-4fe0-a276-5043b064a3a8",
            " has pipeline 'default', version 1",
            "Instrument: TEST1",
            " version is '1'",
            " has configuration 'Old configuration' uuid=6ad5dc90-6b15-43b7-abb5-b07340e19f41",
            " has configuration 'Default configuration' uuid=225fcaf2-7f6f-49cc-972a-70fd0aee8e96",
            " has datamodel 'numina.datamodel.DataModel'",
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


@pytest.mark.parametrize("drpsfunc, expected",[
    (mockreturn0, expecte0),
    (mockreturn1, expecte1),
    (mockreturn2, expecte2)
])
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
