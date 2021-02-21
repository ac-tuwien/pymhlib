#!/usr/bin/env python3
"""Calculate grouped basic statistics for one or two DataFrames/TSV files obtained e.g. from multi-run-summary.

The input data are either given via stdin or in one or two files provided
as parameters. If two tsv files are given, they are assumed to be results
from two different algorithms on the same instances, and they are compared
including a Wilcoxon rank sum test.

Important: For the aggregation to work correctly, adapt in particular
below definitions of categ, categ2 and categbase according to your
conventions for the filenames encoding instance and run information.
"""

import re
import sys
import math

import configargparse as argparse
import numpy
import pandas as pd
import scipy.stats


def categ(x):
    """Determine the category name to aggregate over from the given file name.

    For aggregating a single table of raw data, return category name for a given file name.
    """
    x[:5]  # re.sub(r"^(.*)$", r"\1", x)
    # re.sub(r"^(.*)/(T.*)-(.*)_(.*).res",r"\1/\2-\3",x)
    # return re.sub(r".*[lcs|lcps]_(\d+)_(\d+)_(\d+)\.(\d+)(\.out)", r"\1/\2-\3",x)
    #return re.sub(r"([^/#_]*)(_.*_)?([^/#_]*)(__\d+)?([^/#_]*)\.out",
    #    r"\1\3\5",x)


def categ2(x):
    """Extract category name from file name for aggregating two tables.

    For aggregating two tables corresponding to two different summary files,
    extract category name from the given file names that shall be compared.
    """
    x[3:]
    # re.sub(r"^(.*)/(T.*)-(.*)_(.*).res",r"\2-\3")
    # return re.sub(r"^.*[lcs|lcps]_(\d+)_(\d+)_(\d+)\.(\d+)(\.out)", r"\1/\2-\3", x)
    #return re.sub(r"^.*/([^/#_]*)(_.*_)?([^/#_]*)(#\d+)?([^/#_]*)\.out", r"\1\2\3\4\5",x)


def categbase(x):
    """Basename of filename on which two tables should be merged.

    For aggregating two tables corresponding to two different
    configurations that shall be compared, return detailed name of run
    (basename) that should match a corresponding one of the other configuration.
    """
    x[3:]
    # re.sub(r"^.*/(T.*)-(.*)_(.*).res",r"\1-\2-\3",x)
    # re.sub(r"^.*[lcs|lcps]_(\d+)_(\d+)_(\d+)\.(\d+)(\.out)", r"\1_\2_\3.\4\5", x)
    # re.sub(r"^.*/([^/#_]*)(_.*_)?([^/#_]*)(#\\d+)?([^/#_]*)\\.out", r"\1_\2_\3.\4\5",x)

def print_table_context():
    """Set display options forprinting tables unabbreviated.
    """
    return pd.option_context("display.width",100000,
                             "display.max_rows",1000,
                             "display.max_colwidth",10000,
                             "display.precision",8)

#--------------------------------------------------------------------------------
# General helper functions

def geometric_mean(x, shift=0):
    """Calculates geometric mean with shift parameter."""
    return math.exp(numpy.mean(math.log(x + shift))) - shift

#-------------------------------------------------------------------------
# Aggregation of one summary data frame

def aggregate(raw_data: pd.DataFrame, categ_factor=categ):
    """Determine aggregated results for one summary data frame."""
    raw_data["cat"] = raw_data.apply(lambda row: categ_factor(row["file"]), axis=1)
    # raw_data["gap"] = raw_data.apply(lambda row: (row["ub"]-row["obj"])/row["ub"], axis=1)
    grp = raw_data.groupby("cat")
    aggregated=pd.DataFrame({"runs":grp["obj"].size()})
    aggregated["obj_mean"]=grp["obj"].mean()
    aggregated["obj_sd"]=grp["obj"].std()
    return aggregated
    # aggregated = pd.DataFrame({"runs":grp["obj"].size(),
    #                           "obj_mean":grp["obj"].mean(),
    #                           "obj_sd":grp["obj"].std(),
    #                           "ittot_med":grp["ittot"].median(),
    #                           "ttot_med":grp["ttot"].median(),
    #                           "ub_mean":grp["ub"].mean(),
    #                           "gap_mean":grp["gap"].mean(),
    #                           "tbest_med":grp["tbest"].median(),
    #                           # "tbest_sd":grp["tbest"].std(),
    #                           })
    # return aggregated[["runs","obj_mean","obj_sd","ittot_med","ttot_med",
    #                   "ub_mean","gap_mean","tbest_med"]]


def totalagg(agg):
    """Calculate total values over aggregate data."""
    total = pd.DataFrame({"total": [""],
               "runs": agg["runs"].sum(),
               "obj_mean": agg["obj_mean"].mean(),
               "obj_sd": agg["obj_sd"].mean(),
               # "ittot_med": [agg["ittot_med"].median()],
               # "ttot_med": [agg["ttot_med"].median()],
               # "tbest_med": [agg["tbest_med"].median()],
               # "tbest_sd": agg["tbest"].std(),
            })
    total = total[["total", "runs", "obj_mean", "obj_sd"
                    # "ittot_med", "ttot_med", "ttot_med", "tbest_med"
                 ]]
    total = total.set_index("total")
    total.index.name = None
    return total


def make_index(a):
    """Assumes that the index is a filename and turns it into a multilevel index."""
    idx = {}
    for fn in a.index:
        l = re.findall(r"[a-zA-Z]+\d+",fn)
        for s in l:
            m = re.match(r"([a-zA-Z]+)(\d+)$",s)
            if m.group(1) in idx:
                idx[m.group(1)].append(int(m.group(2)))
            else:
                idx[m.group(1)] = [int(m.group(2))]
    for k in idx:
        a[k] = idx[k]
    a.set_index(list(idx.keys()),inplace=True)
    return a


def roundagg(a):
    """Reasonably round aggregated results for printing."""
    return a.round({'obj_mean': 6, 'obj_sd': 6, 'ittot_med': 1, 'itbest_med': 1,
        'ttot_med': 1, 'tbest_med': 1, 'obj0_mean': 6, 'obj1_mean': 6})


def agg_print(raw_data):
    """Perform aggregation and print results for one summary data frame."""
    aggregated = aggregate(raw_data)
    aggtotal = totalagg(aggregated)
    with print_table_context():
        print(roundagg(aggregated))
        print("\nTotals:")
        print(roundagg(aggtotal))

#-------------------------------------------------------------------------
# Aggregation and comparison of two summary data frames

def one_sided_wilcoxon_test(col1, col2) -> float:
    """Perform one-sided Wilcoxon signed rank-test for the assumption col1 < col2 and return p-value."""
    dif = col1 - col2
    no_ties = len(dif[dif != 0])
    if no_ties < 1:
        return 1.0
    # if (col1==col2).all():
    #     return 3
    # with warnings.catch_warnings():
    #   warnings.simplefilter("ignore")
    _msr, p = scipy.stats.wilcoxon(col1, col2, correction=True, zero_method="wilcox", alternative="less")
    # s,p = scipy.stats.mannwhitneyu(col1,col2,alternative="less")
    # p = p/2
    # if not lessass:
    #     p = 1-p
    return p

stat_test = one_sided_wilcoxon_test


def do_aggregate2(raw: pd.DataFrame, fact: str, criterion):
    """Aggregate results of differences for the given criterion on two merged summary data frames.
    """
    c_diff = criterion+"_diff"
    c_x = criterion+"_x"
    c_y = criterion+"_y"
    raw[c_diff] = raw.apply(lambda row: row[c_x]-row[c_y],axis=1)
    raw["X_less_Y"] = raw.apply(lambda row: int(row[c_x]<row[c_y]), axis=1)
    raw["Y_less_X"] = raw.apply(lambda row: int(row[c_x]>row[c_y]), axis=1)
    raw["X_eq_Y"] = raw.apply(lambda row: int(row[c_x]==row[c_y]), axis=1)
    # rawdata["gap"]=raw.apply(lambda row: (row["ub"]-row["obj"])/row["ub"], axis=1)
    grp = raw.groupby(fact)
    p_X_less_Y = []
    p_Y_less_X = []
    for _g, d in grp:
        p_X_less_Y.append(stat_test(d[c_x],d[c_y]))
        p_Y_less_X.append(stat_test(d[c_y],d[c_x]))
    aggregated = pd.DataFrame({"runs":grp[c_x].size()})
    aggregated["X_"+criterion+"_mean"] = grp[c_x].mean()
    aggregated["Y_"+criterion+"_mean"] = grp[c_y].mean()
    aggregated["diff_mean"] = grp[c_diff].mean()
    aggregated["X_less_Y"] = grp["X_less_Y"].sum()
    aggregated["Y_less_X"] = grp["Y_less_X"].sum()
    aggregated["X_eq_Y"] = grp["X_eq_Y"].sum()
    aggregated["p_X_less_Y"] = p_X_less_Y
    aggregated["p_Y_less_X"] = p_Y_less_X
    return aggregated


def aggregate2(rawdata1: pd.DataFrame, rawdata2: pd.DataFrame, criterion):
    """Determine aggregated results for two summarry data frames.

    This includes statistical tests for significant differences of results for the
    given criterion.
    """
    rawdata1["base"] = rawdata1.apply(lambda row: categbase(row["file"]), axis=1)
    rawdata2["base"] = rawdata2.apply(lambda row: categbase(row["file"]), axis=1)
    raw = pd.merge(rawdata1, rawdata2, on="base", how="outer")
    print(rawdata1)
    raw["class"] = raw.apply(lambda row: categ2(row["file_x"]), axis=1)
    aggregated = do_aggregate2(raw,"class", criterion)
    raw["total"] = raw.apply(lambda row: "total", axis=1)
    aggtotal = do_aggregate2(raw, "total", criterion)
    return {"grouped": aggregated, "total": aggtotal}


def roundagg2(a: pd.DataFrame, criterion):
    """Round aggregated data for two summary data frames for printing."""
    a["X_less_Y"] = a["X_less_Y"].map(int)
    a["Y_less_X"] = a["Y_less_X"].map(int)
    a["X_eq_Y"] = a["X_eq_Y"].map(int)
    a = a.round({"X_"+criterion+"_mean":6, "Y_"+criterion+"_mean":6, "diff_mean":6,
                    "X_less_Y":0, "Y_less_X":0, "X_eq_Y":0, "p_X_less_Y":4, "p_Y_less_X":4})
    return a


def aggregate_and_compare(raw, fact, col_name: str = 'obj', add_total=True, rounded=None):
    """Compare two result columns in merged data frames."""
    cx, cy = col_name + '_x', col_name + '_y'
    raw["X_minus_Y"] = raw.apply(lambda row: row[cx] - row[cy], axis=1)
    raw["X_less_Y"] = raw.apply(lambda row: row[cx] < row[cy], axis=1)
    raw["Y_less_X"] = raw.apply(lambda row: row[cx] > row[cy], axis=1)
    raw["X_eq_Y"] = raw.apply(lambda row: row[cx] == row[cy], axis=1)
    # rawdata["gap"] = raw.apply(lambda row: (row["ub"]-row["obj"])/row["ub"], axis=1)
    grp = raw.groupby(fact)
    p_X_less_Y = {}
    p_Y_less_X = {}
    for g, d in grp:
        p_X_less_Y[g] = one_sided_wilcoxon_test(d[cx], d[cy])
        p_Y_less_X[g] = one_sided_wilcoxon_test(d[cy], d[cx])
    aggregated = pd.DataFrame({"runs": grp[cx].size(),
                               "X_mean": grp[cx].mean(),
                               "Y_mean": grp[cy].mean(),
                               "X_minus_Y_mean": grp['X_minus_Y'].mean(),
                               "X_less_Y": grp["X_less_Y"].sum(),
                               "Y_less_X": grp["Y_less_X"].sum(),
                               "X_eq_Y": grp["X_eq_Y"].sum(),
                               "p_X_less_Y": p_X_less_Y,
                               "p_Y_less_X": p_Y_less_X,
                               })
    aggregated = aggregated[["runs", "X_mean", "Y_mean", "X_minus_Y_mean", "X_less_Y", "Y_less_X", "X_eq_Y",
                             "p_X_less_Y", "p_Y_less_X"]]
    if add_total:
        raw['total'] = 'total'
        agg_total = aggregate_and_compare(raw, 'total', col_name, add_total=False)
        aggregated = pd.concat([aggregated, agg_total])
    if rounded:
        aggregated = round_compared(aggregated, rounded)
    return aggregated


def round_compared(a: pd.DataFrame, digits: int = 2):
    """Rounds aggregated data for two summary data frames for printing."""
    a["X_less_Y"] = a["X_less_Y"].map(int)
    a["Y_less_X"] = a["Y_less_X"].map(int)
    a["X_eq_Y"] = a["XeqY"].map(int)
    a = a.round({"X_mean": digits, "Y_mean": digits, "X_minus_Y_mean": digits,
                 "X_less_Y": 0, "Y_less_X": 0, "X_eq_Y": 0, "p_X_less_Y": 4, "p_Y_less_X": 4})
    return a


def print_sig_diffs(agg2):
    """Print significant differences in aggregated data for two summary data frames."""
    X_winner = sum(agg2["X_less_Y"] > agg2["Y_less_X"])
    Y_winner = sum(agg2["X_less_Y"] < agg2["Y_less_X"])
    gr = agg2["X_less_Y"].size
    print("X is yielding more frequently better results on ", X_winner,
          " groups (", round(X_winner / gr * 100, 2), "%)")
    print("Y is yielding more frequently better results on ", Y_winner,
          " groups (", round(Y_winner / gr * 100, 2), "%)")
    print("\nSignificant differences:")
    sig_X_less_Y = agg2[agg2.p_X_less_Y <= 0.05]
    sig_Y_less_X = agg2[agg2.p_Y_less_X <= 0.05]
    if not sig_X_less_Y.empty:
        print("\np_X_less_Y<=0.05\n", sig_X_less_Y)
    if not sig_Y_less_X.empty:
        print("\np_Y_less_X<=0.05\n", sig_Y_less_X)


def agg2_print(rawdata1: pd.DataFrame, rawdata2: pd.DataFrame, criterion: str):
    """Perform aggregation and print comparative results for two summary DataFrames.
    """
    with print_table_context():
        aggregated = aggregate2(rawdata1,rawdata2,criterion)
        print(aggregated)
        print(roundagg2(pd.concat([aggregated["grouped"],aggregated["total"]]),
                        criterion))
        #print(roundagg2(aggregated["total"]))
        print("")
        print_sig_diffs(roundagg2(pd.concat([aggregated["grouped"],
                                           aggregated["total"]]), criterion))

def main():
    """Main program for calculating grouped basic statistics."""
    parser = argparse.ArgumentParser(
        description="Calculate aggregated statistics for one or two summary files obtained from summary.py",
        config_file_parser_class=argparse.YAMLConfigFileParser,
        default_config_files=['aggregate-results.cfg'])
    parser.add_argument('-t', '--times', action="store_true", default=False,
                        help='Consider total times for proven optimal solutions (10000000 if no opt prove)')
    parser.add_argument("file", nargs="?", help="File from summary.py to be aggregated")
    parser.add_argument("file2", nargs="?", help="Second file from summary.py to be aggregated and compared to")
    parser.add_argument("-c", "--criterion", default="obj", help="Criterion for statistical tests, default: obj")
    # parser.add_argument('-c', '--config', is_config_file=True, help='YAML-config file to be read')
    args = parser.parse_args()

    print(args.file2)

    if not args.file2:
        # process one summary file
        f = args.file if args.file else sys.stdin
        rawdata = pd.read_csv(f, sep='\t')
        # rawdata["obj"] = calculate_obj(rawdata, args)
        agg_print(rawdata)
    else:
        # process and compare two summary files
        rawdata1 = pd.read_csv(args.file, sep='\t')
        # rawdata1["obj"] = calculate_obj(rawdata1, args)
        rawdata2 = pd.read_csv(args.file2, sep='\t')
        # rawdata2["obj"] = calculate_obj(rawdata2, args)
        agg2_print(rawdata1, rawdata2, args.criterion)


if __name__ == "__main__":
    main()
