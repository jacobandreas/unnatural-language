#!/usr/bin/env python3

import os

results = {}
for dataset in os.listdir("."):
    if not os.path.isdir(dataset):
        continue
    results[dataset] = {}
    for experiment in os.listdir(dataset):
        result_line = None
        result_filename = os.path.join(dataset, experiment, "predict.log")
        if not os.path.exists(result_filename):
            continue
        with open(result_filename) as f:
            for line in f:
                if "Stats for iter=1.test" in line:
                    result_line = line.strip()
        if result_line is None:
            continue
        parts = result_line.split()
        correct = parts[3]
        assert "correct" in correct
        score = float(correct.split("=")[1])
        results[dataset][experiment] = score

experiments = sorted(list(results["basketball"].keys()))
datasets = list(sorted(results.keys()))
print("\\documentclass{article} \\begin{document}")
print("\\begin{tabular}{l" + ("c" * len(results) ) + "}")
print("& " + " & ".join(datasets) + " \\\\")
for experiment in experiments:
    line = [experiment]
    for dataset in datasets:
        if experiment not in results[dataset]:
            line.append("")
        else:
            line.append("{:.2f}".format(results[dataset][experiment]))
    print(" & ".join(line) + " \\\\")
print("\\end{tabular}")
print("\\end{document}")
