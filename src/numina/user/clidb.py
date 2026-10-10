#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, the registry of reductions.

``numina db`` works with the registry of reductions (``--db`` or ``file``
in ``[tool.db]``):

* ``numina db ingest`` records observing blocks, without reducing them;
* ``numina db run`` reduces observing blocks by their id;
* ``numina db list`` shows the contents of the registry.
"""

import argparse
import logging
import os
import sys

from .clirun import add_location_options, add_reduction_options, complete_config
from .clirundal import configure_run, run_oblocks
from .helpers import ReductionRequests, create_datamanager, load_observations

_logger = logging.getLogger(__name__)

TASK_STATES = {0: "created", 1: "running", 2: "finished", 3: "failed"}


def register(subparsers, config):
    complete_config(config)
    parser_db = subparsers.add_parser(
        "db",
        help="work with the registry of reductions",
        description="Work with the registry of reductions (--db or file in [tool.db] of the configuration).",
    )
    subdb = parser_db.add_subparsers(title="Targets", dest="db_command", metavar="{ingest,run,list}")
    subdb.required = True

    location = argparse.ArgumentParser(add_help=False)
    add_location_options(location)

    parser_ingest = subdb.add_parser(
        "ingest", parents=[location], help="record observing blocks in the registry, without reducing them"
    )
    parser_ingest.add_argument("obsresult", nargs="+", help="file with observing blocks")
    parser_ingest.set_defaults(command=db_ingest)

    parser_run = subdb.add_parser("run", parents=[location], help="reduce observing blocks of the registry by id")
    add_reduction_options(parser_run)
    parser_run.add_argument(
        "oblock_ids",
        nargs="*",
        metavar="ID",
        help="id of an observing block, by default the blocks of the requests (--request)",
    )
    parser_run.set_defaults(command=db_run, strict_reqs=True)

    parser_list = subdb.add_parser("list", parents=[location], help="show the contents of the registry")
    parser_list.add_argument("collection", choices=["oblocks", "tasks", "results", "products"])
    parser_list.add_argument("--oblock-id", help="only the documents of this observing block")
    parser_list.add_argument("--instrument", help="only the documents of this instrument")
    parser_list.add_argument("--mode", help="only the documents of this observing mode")
    parser_list.set_defaults(command=db_list)

    return parser_db


def open_registry(args, config):
    """The registry of reductions, from the options and the configuration

    Returns None, after printing an error, if there is no registry.
    """
    from numina.dal.registry import Registry

    if args.db:
        filename = args.db
    elif config.has_section("tool.db") and config["tool.db"].get("file"):
        filename = config["tool.db"]["file"]
    else:
        print("numina db: error: no registry of reductions, use --db FILE or file in [tool.db]", file=sys.stderr)
        return None
    basedir = args.basedir or config["tool.run"]["basedir"]
    return Registry(os.path.join(basedir, filename), basedir=basedir)


def db_ingest(args, extra_args, config):
    """Record the observing blocks of the files in the registry"""
    registry = open_registry(args, config)
    if registry is None:
        return 2
    _, loaded_obs = load_observations(args.obsresult)
    for oblock in loaded_obs:
        status = registry.record_oblock(oblock)
        print(f"{status} oblock_id={oblock['id']}")
    return 0


def db_run(args, extra_args, config):
    """Reduce observing blocks of the registry"""
    configure_run(args, config)
    datamanager = create_datamanager(config, args.reqs, extra_args.extra_control, profile_path_extra=args.profilepath)
    if datamanager.registry is None:
        print("numina db: error: no registry of reductions, use --db FILE or file in [tool.db]", file=sys.stderr)
        return 2
    requests = None if args.request is None else ReductionRequests.load(args.request)
    names = args.oblock_ids or ([] if requests is None else requests.ids())
    if not names:
        print("numina db: error: no observing blocks, give their ids or a request file with ids", file=sys.stderr)
        return 2
    # The ids of the command line are strings, the ids
    # of the observing blocks can be integers
    known = {str(doc["oblock_id"]): doc["oblock_id"] for doc in datamanager.registry.oblocks()}
    oblock_ids = []
    for name in map(str, names):
        if name not in known:
            print(f"numina db: error: oblock_id={name} is not in the registry", file=sys.stderr)
            return 2
        oblock_ids.append(known[name])
    run_oblocks(datamanager, oblock_ids, args, config, requests=requests)
    return 0


def _table(registry, collection):
    """Header and rows with the main fields of the documents of a collection"""
    if collection == "oblocks":
        header = ["oblock_id", "instrument", "mode", "frames"]
        rows = []
        for doc in registry.oblocks():
            ob = doc["definition"]
            frames = ob.get("frames", ob.get("images", []))
            rows.append([ob["id"], ob.get("instrument"), ob.get("mode"), len(frames)])
    elif collection == "tasks":
        header = ["id", "oblock_id", "instrument", "mode", "state", "time_end"]
        rows = []
        for doc in registry.tasks():
            runinfo = doc.get("request_runinfo") or {}
            state = TASK_STATES.get(doc.get("state"), doc.get("state"))
            rows.append(
                [
                    doc["id"],
                    doc["oblock_id"],
                    runinfo.get("instrument"),
                    runinfo.get("mode"),
                    state,
                    doc.get("time_end"),
                ]
            )
    elif collection == "results":
        header = ["id", "uuid", "task_id", "oblock_id", "instrument", "mode", "qc", "time_create"]
        rows = [[doc.get(field) for field in header] for doc in registry.results()]
    else:
        header = ["id", "uuid", "type", "oblock_id", "instrument", "qc", "time_obs", "content"]
        rows = [[doc.get(field) for field in header] for doc in registry.products()]
    return header, rows


def db_list(args, extra_args, config):
    """Print a table with the documents of a collection of the registry"""
    registry = open_registry(args, config)
    if registry is None:
        return 2
    header, rows = _table(registry, args.collection)
    filters = {"oblock_id": args.oblock_id, "instrument": args.instrument, "mode": args.mode}
    for key, value in filters.items():
        if value is None:
            continue
        if key not in header:
            print(f"numina db: error: the {args.collection} have no field {key}", file=sys.stderr)
            return 2
        idx = header.index(key)
        rows = [row for row in rows if str(row[idx]) == value]

    if not rows:
        print(f"no {args.collection}")
        return 0
    cells = [header] + [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [max(len(cell) for cell in column) for column in zip(*cells)]
    for line in cells:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip())
    return 0
