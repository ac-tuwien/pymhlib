#!/usr/bin/python3.7
"""Summarizes essential information from multiple mhlib algorithm runs found in the respective out and log files.

The information to be extracted from each out-file is specified by the list to_fetch containing tuples,
where the first element is some numeric value indicating the order in which the elements are appearing in each out-file,
the second element is a name, and the third element is a regular expression for extracting the element.
Values >= 100 indicate that also the corresponding log-file needs to be searched and the corresponding elements
are to be extracted from there.

Example for a YAML config file:

fetch:
    '[
    (10, "obj", r"^best obj:\s(\d+\.?\d*)"),
    (30, "ittot", r"^total iterations:\s(\d+\.?\d*)"),
    (20, "itbest", r"^best iteration:\s(\d+\.?\d*)"),
    (50, "ttot", r"^total time \[s\]:\s(\d+\.?\d*)"),
    (40, "tbest", r"best time \[s\]:\s(\d+\.?\d*)"),
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
    (10, 'obj', r'^best obj:\s(\d+\.?\d*)'),
    (30, 'ittot', r'^total iterations:\s(\d+\.?\d*)'),
    (20, 'itbest', r'^best iteration:\s(\d+\.?\d*)'),
    (50, 'ttot', r'^total time \[s\]:\s(\d+\.?\d*)'),
    (40, 'tbest', r'^best time \[s\]:\s(\d+\.?\d*)'),
    (110, 'obj0', r'^0+\s(\d+.?\d*)'),
    (120, 'obj1', r'^0+1\s(\d+.?\d*)'),
]


@dataclass
class Data:
    nr_to_fetch: int
    name: str
    reg_exp: str
    reg_exp_compiled: Any
    values: List


def _parse_file(file: str, fetch, fetch_iter) -> bool:
    """Parse file file, looking for fetch and when found take next fetch from fetch_iter.

    :return: True when all information found, else False
    """
    # print(file)
    with open(file) as f:
        for line in f:
            m = re.match(fetch.reg_exp_compiled, line)
            if m:
                fetch.values.append(m[1])
                try:
                    fetch = next(fetch_iter)
                except StopIteration:
                    return True
    return False


def parse_files(paths, to_fetch):
    """Process list of files/directories."""
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(f for f in glob.glob(path + "**/*.out", recursive=True))
        else:
            files.append(path)
    to_fetch_data = [Data(fetch[0], fetch[1], fetch[2], re.compile(fetch[2]), []) for fetch in to_fetch]
    to_fetch_data_sorted = sorted(to_fetch_data, key=lambda d: d.nr_to_fetch)
    for file in files:
        # process out-file
        fetch_iter = iter(to_fetch_data_sorted)
        fetch = next(fetch_iter)
        completed = _parse_file(file, fetch, fetch_iter)
        if not completed and fetch.nr_to_fetch >= 100:
            # also process corresponding log file
            log_file = re.sub("(.out)$", ".log", file)
            completed = _parse_file(log_file, fetch, fetch_iter)
        if completed:
            df = DataFrame({fetch.name: fetch.values for fetch in to_fetch_data})
            df.insert(0, 'file', files)
            df.set_index('file', inplace=True)
            return df
        else:
            # remove partially extracted information
            length = len(to_fetch_data_sorted[-1].values)
            for fetch in to_fetch_data_sorted:
                del fetch.values[length:]


def main():
    to_fetch = fetch
    parser = p.ArgumentParser(description='Summarize results for multiple mhlib runs from their .out files.',
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
