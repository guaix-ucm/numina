"""Definitions of DRPs for the tests, from their modes and recipes"""

import yaml

CONFIGURATIONS = "tests.support.drps.configs"


def recipe_path(recipe):
    """The import path of a recipe class, a string or a dict are not changed"""
    if isinstance(recipe, type):
        return f"{recipe.__module__}.{recipe.__qualname__}"
    return recipe


def make_drp(recipes=None, *, name="TEST1", pipelines=None, mode_options=None, **extra):
    """A DRP for the tests, in YAML

    Parameters
    ----------
    recipes : dict, optional
        The recipe of each mode, in the pipeline default. A recipe is a
        class, its import path or a dict with class and kwargs.
    name : str
        Name of the instrument.
    pipelines : dict, optional
        The recipes of each mode, by pipeline, instead of recipes.
    mode_options : dict, optional
        Other keys of each mode, such as rawimage or validator.
    extra
        Other keys of the DRP, such as version or calibrations.

    Each mode has a name, a summary and a description made from its key,
    and the instrument configurations are those of tests.support.drps.
    """
    if pipelines is None:
        pipelines = {"default": recipes or {}}
    mode_options = mode_options or {}
    keys = []
    for pipeline in pipelines.values():
        keys.extend(key for key in pipeline if key not in keys)
    modes = []
    for key in keys:
        mode_name = key.capitalize()
        mode = {"key": key, "name": mode_name, "summary": f"{mode_name} mode", "description": f"{mode_name} mode"}
        mode.update(mode_options.get(key, {}))
        modes.append(mode)
    drp = {
        "name": name,
        "configurations": {"path": CONFIGURATIONS},
        "modes": modes,
        "pipelines": {
            pipeline: {"version": 1, "recipes": {key: recipe_path(recipe) for key, recipe in recipes_.items()}}
            for pipeline, recipes_ in pipelines.items()
        },
    }
    drp.update(extra)
    return yaml.safe_dump(drp, sort_keys=False)
