# Swift Autodiff Benchmarking

Compare Swift automatic differentiation performance versus manually written pullback struct examples.

Some code adapted from [@shabalind](https://github.com/shabalind).

---

Run `swift run -c release` to benchmark.

Currently, autodiff-generated code is quite slower than manually written similar Swift code.

```
name                               time   std        iterations
---------------------------------------------------------------
forward pass                        36 ns ± 399.20 %    1000000
gradient (autodiff)                650 ns ± 111.40 %    1000000
gradient (manual pullback structs)  36 ns ± 619.85 %    1000000
```

Run `python godbolt.py` to produce SIL outputs in [`godbolt/`](godbolt).
