#!/usr/bin/python3.7
"""Summarizes essential information from multiple pymhlib algorithm runs found in the respective out and log files.

The information to be extracted from each out-file is specified by the list to_fetch containing tuples,
where the first element is some numeric value indicating the order in which the elements are appearing in each out-file,
the second element is a name, and the third element is a regular expression for extracting the element.
Values >= 100 indicate that also the corresponding log-file needs to be searched and the corresponding elements
are to be extracted from there.

Example for a YAML config file:

fetch:
    '[
    (10, "obj", r"^T best obj:\s(\d+\.?\d*)"),
    (30, "ittot", r"^T total iterations:\s(\d+\.?\d*)"),
    (20, "itbest", r"^T best iteration:\s(\d+\.?\d*)"),
    (50, "ttot", r"^T total time \[s\]:\s(\d+\.?\d*)"),
    (40, "tbest", r"T best time \[s\]:\s(\d+\.?\d*)"),
    ]'
"""

import configargparse as p
import glob
import os
import re
from dataclasses import dataclass
from typing import Any, List
from pandas import DataFrame


"""Configuration of what information to extract from the out/log files."""
fetch = [
    (10, 'obj', r'^T best obj:\s(\d+\.?\d*)'),
    (30, 'ittot', r'^T total iterations:\s(\d+\.?\d*)'),
    (20, 'itbest', r'^T best iteration:\s(\d+\.?\d*)'),
    (50, 'ttot', r'^T total time \[s\]:\s(\d+\.?\d*)'),
    (40, 'tbest', r'^T best time \[s\]:\s(\d+\.?\d*)'),
    (110, 'obj0', r'^I\s+0\s+(\d+.?\d*)'),
    (120, 'obj1', r'^I\s+1\s+(\d+.?\d*)'),
]


@dataclass
class Data:
    nr_to_fetch: int
    name: str
    reg_exp: str
    reg_exp_compiled: Any
    values: List


def _parse_file(file: str, fetch_item, fetch_iter) -> bool:
    """Parse file file, looking for fetch_item and when found take next fetch_item from fetch_iter.

    :return: True when all information found, else False
    """
    # print(file)
    with open(file) as f:
        for line in f:
            m = re.match(fetch_item.reg_exp_compiled, line)
            if m:
                fetch_item.values.append(float(m[1]))
                try:
                    fetch_item = next(fetch_iter)
                except StopIteration:
                    return True
    return False


def parse_files(paths: [List, str], to_fetch=None) -> DataFrame:
    """Process list of files/directories or a single file/directory and return resulting dataframe."""
    if not to_fetch:
        global fetch
        to_fetch = fetch[:-2]
    files = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if os.path.isdir(path):
            files.extend(f for f in glob.glob(path + "**/*.out", recursive=True))
        else:
            files.append(path)
    to_fetch_data = [Data(_fetch[0], _fetch[1], _fetch[2], re.compile(_fetch[2]), []) for _fetch in to_fetch]
    to_fetch_data_sorted = sorted(to_fetch_data, key=lambda d: d.nr_to_fetch)
    fully_parsed_files = []
    for file in files:
        # process out-file
        # print(file)
        fetch_iter = iter(to_fetch_data_sorted)
        fetch_item = next(fetch_iter)
        completed = _parse_file(file, fetch_item, fetch_iter)
        if not completed and fetch_item.nr_to_fetch >= 100:
            # also process corresponding log file
            log_file = re.sub("(.out)$", ".log", file)
            completed = _parse_file(log_file, fetch_item, fetch_iter)
        if not completed:
            # remove partially extracted information
            length = len(to_fetch_data_sorted[-1].values)
            for f in to_fetch_data_sorted:
                del f.values[length:]
        else:
            fully_parsed_files.append(file)
    df = DataFrame({f.name: f.values for f in to_fetch_data})
    df.insert(0, 'file', fully_parsed_files)
    df.set_index('file', inplace=True)
    return df


def main():
    to_fetch = fetch
    parser = p.ArgumentParser(description='Summarize results for multiple pymhlib runs from their .out files.',
                              config_file_parser_class=p.YAMLConfigFileParser)
    parser.add_argument('paths', type=str, nargs='+',
                        help='a .out file or directory (tree) containing .out files')
    parser.add_argument('--log', type=bool, default=False, help='also process corresponding .log files')
    parser.add_argument('--fetch', type=str, default=None,
                        help='list of tuples specifying what information to fetch')
    parser.add_argument('-c', '--config', is_config_file=True, help='YAML-config file to be read')
    parser._default_config_files = ['multi-run-summary.cfg']

    args = parser.parse_args()
    if args.fetch:
        to_fetch = eval(args.fetch)
    else:
        if not args.log:
            del fetch[-2:]
    df = parse_files(args.paths, to_fetch)
    print(df.to_csv(sep='\t'))


if __name__ == '__main__':
    main()
