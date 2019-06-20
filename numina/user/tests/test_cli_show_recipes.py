
import pkgutil

import numina.drps
import numina.drps.drpbase
import numina.core.pipelineload as pload

from ..cli import main


drpdata1 = pkgutil.get_data('numina.drps.tests', 'drptest1.yaml')
drpdata2 = pkgutil.get_data('numina.drps.tests', 'drptest2.yaml')


# def test_show_recipes(capsys, monkeypatch):
def test_show_recipes(capsys, monkeypatch):
    """Test that one instrument is shown"""

    def mockreturn():
        import numina.drps.drpbase
        import numina.core.pipelineload as pload
        drps = {}

        drp1 = pload.drp_load_data('numina', drpdata1)
        drps[drp1.name] = drp1
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ["Recipe: numina.core.utils.AlwaysFailRecipe",
                " summary: A Recipe that always fails.",
                " instrument: TEST1",
                "  pipeline: default",
                "  obs mode: fail",
                " requirements:",
                "",
                "Recipe: numina.tests.recipes.BiasRecipe",
                " instrument: TEST1",
                "  pipeline: default",
                "  obs mode: bias",
                " requirements:",
                "",
                "Recipe: numina.tests.recipes.DarkRecipe",
                " instrument: TEST1",
                "  pipeline: default",
                "  obs mode: dark",
                " requirements:",
                "",
                "Recipe: numina.tests.recipes.ImageRecipe",
                " instrument: TEST1",
                "  pipeline: default",
                "  obs mode: image",
                " requirements:",
                "  obresult type='ObservationResultType()' [Observation Result]",
                "  master_bias type='MasterBias' [Master Bias]",
                "  master_dark type='MasterDark' [Master Dark]",
                "",
                ""
                ]

    main(['show-recipes'])
    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    print('NN')
    print(out)
    print(expected)
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

    expected = [u'Recipe: numina.core.utils.AlwaysFailRecipe',
                u' summary: A Recipe that always fails.',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: fail',
                u' requirements:',
                u'',
                u'Recipe: numina.tests.recipes.BiasRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: bias',
                u' requirements:',
                u'',
                u'Recipe: numina.tests.recipes.DarkRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: dark',
                u' requirements:',
                u'',
                u"Recipe: numina.tests.recipes.ImageRecipe",
                u" instrument: TEST1",
                u"  pipeline: default",
                u"  obs mode: image",
                u" requirements:",
                u"  obresult type='ObservationResultType()' [Observation Result]",
                u"  master_bias type='MasterBias' [Master Bias]",
                u"  master_dark type='MasterDark' [Master Dark]",
                u"",
                u'Recipe: numina.tests.recipes.DarkRecipe',
                u' instrument: TEST2',
                u'  pipeline: default',
                u'  obs mode: dark',
                u' requirements:',
                u'',
                u'Recipe: numina.core.utils.AlwaysSuccessRecipe',
                u' summary: A Recipe that always successes.',
                u' instrument: TEST2',
                u'  pipeline: default',
                u'  obs mode: success',
                u' requirements:',
                u'',
                u''
                ]

    main(['show-recipes'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_1_recipe(capsys, monkeypatch):
    """Test that one recipe is shown"""

    def mockreturn():
        drps = {}

        drp1 = pload.drp_load_data('numina', drpdata1)
        drp2 = pload.drp_load_data('numina', drpdata2)
        drps[drp1.name] = drp1
        drps[drp2.name] = drp2
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = [u'Recipe: numina.tests.recipes.BiasRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: bias',
                u' requirements:',
                u'',
                u''
                ]

    main(['show-recipes', 'numina.tests.recipes.BiasRecipe'])
    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_2_recipes(capsys, monkeypatch):
    """Test that two recipes are shown"""

    def mockreturn():
        drps = {}

        drp1 = pload.drp_load_data('numina', drpdata1)
        # drp2 = pload.drp_load_data('numina', drpdata2)
        drps[drp1.name] = drp1
        # drps[drp2.name] = drp2
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = [u'Recipe: numina.tests.recipes.BiasRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: bias',
                u' requirements:',
                u'',
                u'Recipe: numina.tests.recipes.DarkRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: dark',
                u' requirements:',
                u'',
                u''
                ]

    main(
        ['show-recipes',
         'numina.tests.recipes.BiasRecipe',
         'numina.tests.recipes.DarkRecipe'
         ]
    )
    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_2_ins_recipes(capsys, monkeypatch):
    """Test that two recipes are shown, selecting instrument"""

    def mockreturn():
        drps = {}

        drp1 = pload.drp_load_data('numina', drpdata1)
        drp2 = pload.drp_load_data('numina', drpdata2)
        drps[drp1.name] = drp1
        drps[drp2.name] = drp2
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = [u'Recipe: numina.tests.recipes.BiasRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: bias',
                u' requirements:',
                u'',
                u'Recipe: numina.tests.recipes.DarkRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: dark',
                u' requirements:',
                u'',
                u''
                ]

    main(
        ['show-recipes', '-i', 'TEST1',
         'numina.tests.recipes.BiasRecipe',
         'numina.tests.recipes.DarkRecipe'
         ]
    )
    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_mode_name(capsys, monkeypatch):
    """Test that one recipe by mode name is shown"""

    def mockreturn():
        drps = {}

        drp1 = pload.drp_load_data('numina', drpdata1)
        # drp2 = pload.drp_load_data('numina', drpdata2)
        drps[drp1.name] = drp1
        # drps[drp2.name] = drp2
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = [u'Recipe: numina.tests.recipes.BiasRecipe',
                u' instrument: TEST1',
                u'  pipeline: default',
                u'  obs mode: bias',
                u' requirements:',
                u'',
                u''
                ]

    main(['show-recipes', '--mode', 'bias'])
    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_recipes_no_instruments(capsys, monkeypatch):
    """Test that no instruments are shown"""

    def mockreturn():
        return numina.drps.drpbase.DrpGeneric()

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ""

    main(['show-recipes'])

    out, err = capsys.readouterr()

    assert out == expected


def test_show_recipes_2_instruments_select_no(capsys, monkeypatch):
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

    main(['show-recipes', '-i', 'TEST3'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected


def test_show_recipes_no_instruments_select_no(capsys, monkeypatch):
    """Test that two instruments are shown"""

    def mockreturn():
        drps = {}
        return numina.drps.drpbase.DrpGeneric(drps)

    monkeypatch.setattr(numina.drps, "get_system_drps", mockreturn)

    expected = ["No instrument named: TEST3",
                ""
                ]

    main(['show-recipes', '-i', 'TEST3'])

    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected
