#
# Copyright 2019-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import os
import shutil
import tempfile

import pytest

import numina.util.context as ctx
from numina.util.jsonencoder import ExtEncoder


class ResultCompPlugin(object):
    """Plugin to compare the results fo reductions"""
    def __init__(self, config, reference_dir=None, generate_dir=None):
        self.config = config
        self.reference_dir = reference_dir
        self.generate_dir = generate_dir

    def pytest_runtest_setup(self, item):
        import functools
        compare = item.get_closest_marker('result_compare')

        if compare is None:
            return

        original = item.function

        atol = compare.kwargs.get('atol', 0.)
        rtol = compare.kwargs.get('rtol', 1e-7)

        @functools.wraps(item.function)
        def item_function_wrapper(*args, **kwargs):

            reference_dir = compare.kwargs.get('reference_dir', None)
            if reference_dir is None:
                if self.reference_dir is None:
                    reference_dir = os.path.join(os.path.dirname(item.fspath.strpath), 'reference')
                else:
                    reference_dir = self.reference_dir
            else:
                if not reference_dir.startswith(('http://', 'https://')):
                    reference_dir = os.path.join(os.path.dirname(item.fspath.strpath), reference_dir)

            baseline_remote = reference_dir.startswith('http')

            # Run test and get result object
            import inspect
            if inspect.ismethod(original):  # method
                result = original(*args[1:], **kwargs)
            else:  # function
                result = original(*args, **kwargs)

            # Task or result...
            destination = compare.kwargs.get('destination', None)
            if destination is None:
                destination = item.name
                destination = destination.replace('[', '_').replace(']', '_')

            if self.generate_dir is None:

                # Save the result
                result_dir = tempfile.mkdtemp()
                with ctx.working_directory(result_dir):
                    manifest = generate_manifest(result)

                    import json
                    with open('result.json', 'w') as fd:
                        json.dump(manifest, fd, indent=2, cls=ExtEncoder)

                if baseline_remote:
                    # baseline_file_ref = _download_file(reference_dir + filename)
                    raise NotImplementedError
                else:
                    baseline_file_ref = os.path.abspath(
                        os.path.join(os.path.dirname(item.fspath.strpath), reference_dir, destination)
                    )

                if not os.path.exists(baseline_file_ref):
                    exmsg = "File not found for comparison test\nGenerated file:\t{test}\nThis is expected for new tests."
                    raise Exception(exmsg.format(test=destination))

                # Compare my result with something else
                identical, msg = compare_result_dirs(baseline_file_ref, result_dir, atol=atol, rtol=rtol)

                if identical:
                    shutil.rmtree(result_dir)
                else:
                    raise Exception(msg)
            else:
                # Generating
                generate_detail = os.path.join(self.generate_dir, destination)
                if not os.path.exists(generate_detail):
                    os.makedirs(generate_detail)

                with ctx.working_directory(generate_detail):

                    manifest = generate_manifest(result)

                    import json
                    with open('result.json', 'w') as fd:
                        json.dump(manifest, fd)

                # Write something in destination...
                pytest.skip("skipping test, since generating data")

        # Override with wrapped version
        if item.cls is not None:
            setattr(item.cls, item.function.__name__, item_function_wrapper)
        else:
            item.obj = item_function_wrapper


def compare_eq_sequence(left, right):
    explanation = []
    for i in range(min(len(left), len(right))):
        if left[i] != right[i]:
            explanation += [f"At index {i} diff: {left[i]!r} != {right[i]!r}"]
            break
    if len(left) > len(right):
        explanation += [f"Left contains more items, first extra item: {left[len(right)]}"
        ]
    elif len(left) < len(right):
        explanation += [f"Right contains more items, first extra item: {right[len(left)]}"
        ]
    return explanation


def compare_result_dirs(resdir1, resdir2, atol=0.0, rtol=1e-7):
    # First is reference, second is generated
    # print('compare contents of', resdir1, resdir2)
    # Compare dir contents
    cont1 = sorted(os.listdir(resdir1))
    cont2 = sorted(os.listdir(resdir2))

    # as assert cont1 == cont2
    if cont1 != cont2:
        exp = compare_eq_sequence(cont1, cont2)
        return False, exp

    for fname in cont1:
        fname1 = os.path.join(resdir1, fname)
        fname2 = os.path.join(resdir2, fname)
        part, ext = os.path.splitext(fname)
        # Check extensions
        if ext == '.fits':
            from astropy.io.fits.diff import FITSDiff
            ignore_keywords = [
                'HISTORY', 'UUID','NUMXVER'
            ]
            diff = FITSDiff(
                fname1, fname2, rtol=rtol, atol=atol,
                ignore_keywords=ignore_keywords
            )
            return diff.identical, diff.report()
        elif ext == '.json':
            print('json comparator, not implemented')
        else:
            print('ignoring ', ext, 'file')
    return True, "ok"


def generate_manifest(recipe_result):
    import numina.store

    saveres = dict(values={})
    saveres_v = saveres['values']
    for key, prod in recipe_result.stored().items():
        val = getattr(recipe_result, key)
        saveres_v[key] = numina.store.dump(prod.type, val, prod.dest)

    saveres['qc'] = recipe_result.qc.name
    saveres['uuid'] = str(recipe_result.uuid)

    return saveres
