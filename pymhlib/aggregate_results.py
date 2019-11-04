#!/usr/bin/python3.7
"""Calculate grouped basic statistics for one or two dataframes/TSV files obtained e.g. from multi-run-summary.

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
import warnings

import configargparse as argparse
import math
import numpy as np
import pandas as pd
import scipy.stats


def categ(x):
    """Determine the category name to aggregate over from the given file name.
    
    For aggregating a single table of raw data,  
    return category name for a given file name.
    """
    # re.sub(r"^(.*)/(T.*)-(.*)_(.*).res",r"\1/\2-\3",x)
    return re.sub(r".*[lcs|lcps]_(\d+)_(\d+)_(\d+)\.(\d+)(\.out)", 
        r"\1/\2-\3",x)
    #return re.sub(r"([^/#_]*)(_.*_)?([^/#_]*)(__\d+)?([^/#_]*)\.out",
    #    r"\1\3\5",x)


def categ2(x):
    """For aggregating two tables corresponding to two different summary files,
    extract category name from the given file names that shall be compared.
    """
    # re.sub(r"^(.*)/(T.*)-(.*)_(.*).res",r"\2-\3")
    return re.sub(r"^.*[lcs|lcps]_(\d+)_(\d+)_(\d+)\.(\d+)(\.out)",
           r"\1/\2-\3",x)
    #return re.sub(r"^.*/([^/#_]*)(_.*_)?([^/#_]*)(#\d+)?([^/#_]*)\.out",
    #       r"\1\2\3\4\5",x)


def categbase(x):
    """For aggregating two tables corresponding to two different 
    configurations that shall be compared, return detailed name of run 
    (basename) that should match a corresponding one of the other configuration.
    """
    # re.sub(r"^.*/(T.*)-(.*)_(.*).res",r"\1-\2-\3",x)
    return re.sub(r"^.*[lcs|lcps]_(\d+)_(\d+)_(\d+)\.(\d+)(\.out)",
           r"\1_\2_\3.\4\5",x)
    # return re.sub(r"^.*/([^/#_]*)(_.*_)?([^/#_]*)(#\\d+)?([^/#_]*)\\.out",
    #       r"\1_\2_\3.\4\5",x)


# Set display options for output
pd.options.display.width = 10000
pd.options.display.max_rows = None
pd.options.display.precision = 8
# pd.set_eng_float_format(accuracy=8)


def geometric_mean(x, shift=0):
    """Calculates geometric mean with shift parameter."""
    return math.exp(math.mean(math.log(x + shift))) - shift


def calculateObj(rawdata, args):
    if args.times:
        return (rawdata["obj"] == rawdata["UB"]) * rawdata["ttot"] + (
                rawdata["obj"] != rawdata["UB"]) * 100000000
    else:
        return rawdata["obj"]


def aggregate(rawdata, categfactor=categ):
    """Determine aggregated results for one summary data frame."""
    rawdata["cat"] = rawdata.apply(lambda row: categfactor(row["file"]), axis=1)
    rawdata["gap"] = rawdata.apply(lambda row: (row["ub"]-row["obj"])/row["ub"], axis=1)
    grp = rawdata.groupby("cat")
    aggregated = grp.agg({"obj": [size, mean, std],
                          "ittot": median, "ttot": median, "ub":mean,
                          "gap": mean, "tbest": median})[["obj", "ittot", "ttot", "ub"]]
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


def aggregatemip(rawdata, categfactor=categ):
    """Determine aggregated results for one summary data frame for MIP results."""
    rawdata["cat"] = rawdata.apply(lambda row: categfactor(row["file"]), axis=1)
    rawdata["gap"] = rawdata.apply(lambda row: (row["Upper_bound"] -
                                                row["Lower_bound"]) / row["Upper_bound"], axis=1)
    grp = rawdata.groupby("cat")
    aggregated = pd.DataFrame({"runs": grp["obj"].size(),
                               "ub_mean": grp["Upper_bound"].mean(),
                               "ub_sd": grp["Upper_bound"].std(),
                               "lb_mean": grp["Lower_bound"].mean(),
                               "lb_sd": grp["Lower_bound"].std(),
                               "ttot_med": grp["ttot"].median(),
                               "gap_mean": grp["gap"].mean(),
                               # "tbest_med": grp["tbest"].med(),
                               # "tbest_sd": grp["tbest"].std(),
                               })
    return aggregated[["runs", "ub_mean", "ub_sd", "lb_mean", "ttot_med", "gap_mean"]]


def totalagg(agg):
    """Calculate total values over aggregate data.
    """
    total = pd.DataFrame({"total": [""],
               "runs": [agg["runs"].sum()],
               "obj_mean": [agg["obj_mean"].mean()],
               # "obj_sd": agg["obj_sd"].mean(),
               "ittot_med": [agg["ittot_med"].median()],
               "ttot_med": [agg["ttot_med"].median()],
               "tbest_med": [agg["tbest_med"].median()],
               # "tbest_sd": agg["tbest"].std(),
            })
    total = total[["total", "runs", "obj_mean", "ittot_med", "ttot_med", "ttot_med", "tbest_med"]]
    total = total.set_index("total")
    total.index.name = None
    return total


def roundagg(a):
    """Reasonably round aggregated results for printing."""
    return a.round({'obj_mean': 6, 'obj_sd': 6, 'ittot_med': 1, 'itbest_med': 1,
        'ttot_med': 1, 'tbest_med': 1, 'obj0_mean': 6, 'obj1_mean': 6})


def roundaggmip(a):
    """Reasonably round aggregated MIP results for printing."""
    return a.round({'ub_mean': 6, 'ub_sd': 6, 'lb_mean': 6, 'lb_sd': 6, 'ttot_med': 1, 'gap_mean': 1})


def agg_print(rawdata):
    """Perform aggregation and print results for one summary data frame."""
    aggregated = aggregate(rawdata)
    aggtotal = totalagg(aggregated)
    print(roundagg(aggregated))
    print("\nTotals:")
    print(roundagg(aggtotal))


def one_sided_wilcoxon_test(col1, col2) -> float:
    """Perform one-sided Wilcoxon signed rank-test for the assumption col1 < col2 and return p-value."""
    dif = col1 - col2
    no_ties = len(dif[dif != 0])
    if no_ties < 1:
        return float(1)
    # if (col1==col2).all():
    #     return 3
    # with warnings.catch_warnings():
    #   warnings.simplefilter("ignore")
    msr, p = scipy.stats.wilcoxon(col1, col2, correction=True, zero_method="wilcox", alternative="less")
    # s,p = scipy.stats.mannwhitneyu(col1,col2,alternative="less")
    return p


def aggregate_and_compare(raw, fact, col_name: str = 'obj', add_total=True, rounded=None):
    """Compare two result columns in merged data frames."""
    cx, cy = col_name + '_x', col_name + '_y'
    raw["XminusY"] = raw.apply(lambda row: row[cx] - row[cy], axis=1)
    raw["XlessY"] = raw.apply(lambda row: row[cx] < row[cy], axis=1)
    raw["YlessX"] = raw.apply(lambda row: row[cx] > row[cy], axis=1)
    raw["XeqY"] = raw.apply(lambda row: row[cx] == row[cy], axis=1)
    # rawdata["gap"] = raw.apply(lambda row: (row["ub"]-row["obj"])/row["ub"], axis=1)
    grp = raw.groupby(fact)
    p_XlessY = {}
    p_YlessX = {}
    for g, d in grp:
        p_XlessY[g] = one_sided_wilcoxon_test(d[cx], d[cy])
        p_YlessX[g] = one_sided_wilcoxon_test(d[cy], d[cx])
    aggregated = pd.DataFrame({"runs": grp[cx].size(),
                               "X_mean": grp[cx].mean(),
                               "Y_mean": grp[cy].mean(),
                               "XminusY_mean": grp['XminusY'].mean(),
                               "XlessY": grp["XlessY"].sum(),
                               "YlessX": grp["YlessX"].sum(),
                               "XeqY": grp["XeqY"].sum(),
                               "p_XlessY": p_XlessY,
                               "p_YlessX": p_YlessX,
                               })
    aggregated = aggregated[["runs", "X_mean", "Y_mean", "XminusY_mean", "XlessY", "YlessX", "XeqY",
                             "p_XlessY", "p_YlessX"]]
    if add_total:
        raw['total'] = 'total'
        agg_total = aggregate_and_compare(raw, 'total', col_name, add_total=False)
        aggregated = pd.concat([aggregated, agg_total])
    if rounded:
        aggregated = round_compared(aggregated, rounded)
    return aggregated


def aggregate2(rawdata1, rawdata2):
    """Determine aggregated results for two summary data frames including comparison of results."""
    rawdata1["base"] = rawdata1.apply(lambda row: categbase(row["file"]), axis=1)
    rawdata2["base"] = rawdata2.apply(lambda row: categbase(row["file"]), axis=1)
    raw = pd.merge(rawdata1, rawdata2, on="base", how="outer")
    raw["class"] = raw.apply(lambda row: categ2(row["file_x"]), axis=1)
    aggregated = doaggregate2(raw, "class")
    raw["total"] = raw.apply(lambda row: "total", axis=1)
    aggtotal = doaggregate2(raw, "total")
    return {"grouped": aggregated, "total": aggtotal}


def round_compared(a: pd.DataFrame, digits: int = 2):
    """Rounds aggregated data for two summary data frames for printing."""
    a["XlessY"] = a["XlessY"].map(lambda x: int(x))
    a["YlessX"] = a["YlessX"].map(lambda x: int(x))
    a["XeqY"] = a["XeqY"].map(lambda x: int(x))
    a = a.round({"X_mean": digits, "Y_mean": digits, "XminusY_mean": digits,
                 "XlessY": 0, "YlessX": 0, "XeqY": 0, "p_XlessY": 4, "p_YlessX": 4})
    return a


def printsigdiffs(agg2):
    """Print significant differences in aggregated data for two summary
    data frames.
    """
    Xwinner = sum(agg2["XlessY"] > agg2["YlessX"])
    Ywinner = sum(agg2["XlessY"] < agg2["YlessX"])
    gr = agg2["XlessY"].size
    print("X is yielding more frequently better results on ", Xwinner,
          " groups (", round(Xwinner / gr * 100, 2), "%)")
    print("Y is yielding more frequently better results on ", Ywinner,
          " groups (", round(Ywinner / gr * 100, 2), "%)")
    print("\nSignificant differences:")
    sigXlessY = agg2[agg2.p_XlessY <= 0.05]
    sigYlessX = agg2[agg2.p_YlessX <= 0.05]
    if not sigXlessY.empty:
        print("\np_XlessY<=0.05\n", sigXlessY)
    if not sigYlessX.empty:
        print("\np_YlessX<=0.05\n", sigYlessX)


def agg2_print(rawdata1, rawdata2):
    """Perform aggregation and print comparative results for two summary dataframes."""
    aggregated = aggregate2(rawdata1, rawdata2)
    print(roundagg2(pd.concat([aggregated["grouped"], aggregated["total"]])))
    # print(roundagg2(aggregated["total"]))
    print("")
    printsigdiffs(roundagg2(pd.concat([aggregated["grouped"], aggregated["total"]])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate aggregated statistics for one or two summary files obtained from summary.py",
        config_file_parser_class=p.YAMLConfigFileParser)
    parser.add_argument('-t', '--times', action="store_true", default=False,
                        help='Consider total times for proven optimal solutions (10000000 if no opt prove)')
    parser.add_argument("file", nargs="?", help="File from summary.py to be aggregated")
    parser.add_argument("file2", nargs="?", help="Second file from summary.py to be aggregated and compared to")
    parser.add_argument('-c', '--config', is_config_file=True, help='YAML-config file to be read')
    parser._default_config_files = ['aggregate-results.cfg']
    args = parser.parse_args()

    print(args.file2)

    if not args.file2:
        # process one summary file
        f = args.file if args.file else sys.stdin
        rawdata = pd.read_csv(f, sep='\t')
        rawdata["obj"] = calculateObj(rawdata, args)
        agg_print(rawdata)
    else:
        # process and compare two summary files
        rawdata1 = pd.read_csv(args.file, sep='\t')
        rawdata1["obj"] = calculateObj(rawdata1, args)
        rawdata2 = pd.read_csv(args.file2, sep='\t')
        rawdata2["obj"] = calculateObj(rawdata2, args)
        agg2_print(rawdata1, rawdata2)
