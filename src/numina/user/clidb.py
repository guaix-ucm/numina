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
* ``numina db list`` shows the contents of the registry;
* ``numina db priority`` sets the priority of a product.
"""

import argparse
import json
import logging
import os
import sys

from .clirun import add_location_options, add_reduction_options, complete_config
from .clirundal import configure_run, run_oblocks
from .helpers import ReductionRequests, create_datamanager, load_observations

_logger = logging.getLogger(__name__)

TASK_STATES = {0: "created", 1: "running", 2: "finished", 3: "failed"}


def register(subparsers, config):
    """Add the ``db`` command and its subcommands to the parser."""
    complete_config(config)
    parser_db = subparsers.add_parser(
        "db",
        help="work with the registry of reductions",
        description="Work with the registry of reductions (--db or file in [tool.db] of the configuration).",
    )
    subdb = parser_db.add_subparsers(title="Targets", dest="db_command", metavar="{ingest,run,list,priority}")
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
    parser_list.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="only the products with this tag, can be repeated",
    )
    output = parser_list.add_mutually_exclusive_group()
    output.add_argument("--tags", action="store_true", help="show the tags of the products, below each one")
    output.add_argument("--json", action="store_true", help="print the documents of the registry in JSON")
    parser_list.set_defaults(command=db_list)

    parser_priority = subdb.add_parser(
        "priority",
        parents=[location],
        help="set the priority of a product in the automatic selection",
        description="Set the priority of a product. The automatic selection of products uses "
        "the product with the highest priority (0 by default), and then the one observed "
        "closest to the observing block.",
    )
    parser_priority.add_argument("product", help="id or uuid (can be abbreviated) of the product")
    parser_priority.add_argument("priority", type=int, help="the priority, an integer")
    parser_priority.set_defaults(command=db_priority)

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
    """Header, rows with the main fields, and documents of a collection"""
    if collection == "oblocks":
        header = ["oblock_id", "instrument", "mode", "frames"]
        docs = registry.oblocks()
        rows = []
        for doc in docs:
            ob = doc["definition"]
            frames = ob.get("frames", ob.get("images", []))
            rows.append([ob["id"], ob.get("instrument"), ob.get("mode"), len(frames)])
    elif collection == "tasks":
        header = ["id", "oblock_id", "instrument", "mode", "state", "time_end"]
        docs = registry.tasks()
        rows = []
        for doc in docs:
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
        docs = registry.results()
        rows = [[doc.get(field) for field in header] for doc in docs]
    else:
        header = ["id", "uuid", "type", "oblock_id", "instrument", "qc", "priority", "time_obs", "content"]
        docs = registry.products()
        rows = [[doc.get(field, 0 if field == "priority" else None) for field in header] for doc in docs]
    return header, rows, docs


def _cell(field, value):
    """The text of a value in the table: the uuids abbreviated, the times in seconds"""
    from numina.dal.registry import UUID_PREFIX_LENGTH

    if value is None:
        return ""
    text = str(value)
    if field == "uuid":
        return text[:UUID_PREFIX_LENGTH]
    if field.startswith("time_"):
        return text[:19]
    return text


def _tags_line(doc):
    """Return the tags of a product as ``KEY=VALUE`` pairs, sorted by key."""
    tags = doc.get("tags") or {}
    if not tags:
        return "(no tags)"
    return " ".join(f"{key}={tags[key]}" for key in sorted(tags))


def product_key(value):
    """The id (an integer) or the uuid of a product, from the command line

    A string of digits shorter than an abbreviated uuid is an id.
    """
    from numina.dal.registry import UUID_PREFIX_LENGTH

    if value.isdigit() and len(value) < UUID_PREFIX_LENGTH:
        return int(value)
    return value


def db_priority(args, extra_args, config):
    """Set the priority of a product of the registry"""
    registry = open_registry(args, config)
    if registry is None:
        return 2
    try:
        product_id = registry.set_priority(product_key(args.product), args.priority)
    except ValueError as error:
        print(f"numina db: error: {error}", file=sys.stderr)
        return 2
    print(f"product {product_id} has priority {args.priority}")
    return 0


def db_list(args, extra_args, config):
    """Print the documents of a collection of the registry, in a table or in JSON"""
    registry = open_registry(args, config)
    if registry is None:
        return 2
    header, rows, docs = _table(registry, args.collection)
    selected = list(zip(rows, docs))
    filters = {"oblock_id": args.oblock_id, "instrument": args.instrument, "mode": args.mode}
    for key, value in filters.items():
        if value is None:
            continue
        if key not in header:
            print(f"numina db: error: the {args.collection} have no field {key}", file=sys.stderr)
            return 2
        idx = header.index(key)
        selected = [(row, doc) for row, doc in selected if str(row[idx]) == value]
    if (args.tag or args.tags) and args.collection != "products":
        print("numina db: error: only the products have tags", file=sys.stderr)
        return 2
    for tag in args.tag:
        key, sep, value = tag.partition("=")
        if not sep:
            print(f"numina db: error: the tag {tag} is not KEY=VALUE", file=sys.stderr)
            return 2
        selected = [
            (row, doc) for row, doc in selected if key in (doc.get("tags") or {}) and str(doc["tags"][key]) == value
        ]

    if args.json:
        print(json.dumps([doc for _, doc in selected], indent=2, default=str))
        return 0
    if not selected:
        print(f"no {args.collection}")
        return 0
    cells = [header] + [[_cell(field, value) for field, value in zip(header, row)] for row, _ in selected]
    widths = [max(len(cell) for cell in column) for column in zip(*cells)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in cells]
    print(lines[0])
    for line, (_, doc) in zip(lines[1:], selected):
        print(line)
        if args.tags:
            print(f"    \u2514\u2500 {_tags_line(doc)}")
    return 0
