#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable
import logging

LOG = logging.getLogger()

parser = argparse.ArgumentParser(
    description='Remove answer cells, reset execution count, and (optionally) clear cell outputs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('input_nb', type=Path)
parser.add_argument('--strip-output', action='store_true')
parser.add_argument('--verbose', '-v', action='count', default=0, dest='verbosity')
output_group = parser.add_mutually_exclusive_group()
output_group.add_argument('--output', type=Path)
output_group.add_argument('--inplace', action='store_true')


def read_json(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf8') as f:
        return json.load(f)


def write_json(file_path: Path, obj: Any):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(obj, f)


def map_over_cells(nb, callback):
    cells = []
    for cell in nb['cells']:
        return_val = callback(cell)
        if return_val is not False:
            cells.append(cell)
    nb['cells'] = cells


def reset_execution_count(nb):
    def reset_count(cell):
        if 'execution_count' in cell:
            cell['execution_count'] = 0
    map_over_cells(nb, reset_count)


def clear_cell_output(nb):
    def clear_output(cell):
        if 'outputs' in cell:
            cell['outputs'] = []
    map_over_cells(nb, clear_output)


def remove_cell_regex(nb, regex: str, re_flags=0):
    regex = re.compile(regex, re_flags)

    def remove_cell(cell):
        found = regex.search("".join(cell['source']))
        if found:
            cell_lines = cell['source']
            cell_header = "".join(cell_lines[:max(5, len(cell_lines) - 1)])
            LOG.debug(f"Removing cell {cell_header}")
        keep = not found
        return keep

    map_over_cells(nb, remove_cell)


def main(args):
    log_levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG
    ]

    logging.basicConfig()
    LOG.setLevel(log_levels[min(args.verbosity, len(log_levels) - 1)])
    nb = read_json(args.input_nb)
    reset_execution_count(nb)
    if args.strip_output:
        clear_cell_output(nb)
    remove_cell_regex(nb, r'.*#\sANSWER\s#.*', re_flags=re.IGNORECASE)
    if args.inplace:
        write_json(args.input_nb, nb)
    else:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, nb)


if __name__ == '__main__':
    main(parser.parse_args())
