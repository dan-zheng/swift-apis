# Some code adapted from https://github.com/shabalind.

import re
import os
import sys
import subprocess as subp

OUTPUT_DIRECTORY = "godbolt"


def sil_function_pattern(name):
    return "^sil.*{name}.*{{(.|[\n])*}} // end sil function '{name}'".format(name=name)


def run_emit_sil(f):
    base = os.path.basename(f).rstrip(".swift")
    try:
        output = subp.check_output(["swiftc", "-O", "-emit-sil", f], stderr=subp.STDOUT)
        with open(OUTPUT_DIRECTORY + "/" + base + ".sil", "w") as w:
            w.write(output)
    except subp.CalledProcessError as err:
        print("Failed to emit sil for: " + f)


def extract_sil_functions(f, function_names):
    assert f.endswith(".sil")
    base = os.path.basename(f).rstrip(".sil")
    with open(f) as r:
        contents = r.read()
        for name in function_names:
            pattern = sil_function_pattern(name)
            match = re.search(pattern, contents, re.MULTILINE)
            try:
                with open(
                    OUTPUT_DIRECTORY + "/" + base + "." + name + ".sil", "w"
                ) as w:
                    w.write(match.group(0))
            except Exception as e:
                print("Failed to extract function {} in file {}: {}".format(name, f, e))


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    run_emit_sil("Sources/AutoDiffBenchmark/Example.swift")
    extract_sil_functions(
        OUTPUT_DIRECTORY + "/Example.sil", ["test_autodiff_gradient_apply", "test_manual_gradient_apply"]
    )
