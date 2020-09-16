# [TF-1302](https://bugs.swift.org/projects/TF/issues/TF-1302)

Simple automatic differentiation benchmark testing a call to `gradient(at:in:)`.

```
name          time   std        iterations
------------------------------------------
forward call   39 ns ± 265.35 %    1000000
gradient call 692 ns ±  59.36 %    1000000
```

[This Gist](https://gist.github.com/dan-zheng/e414e8d80357aa2b703580efe4ac18ac)
has ideas for speeding up the performance of calls to `gradient(at:in:)`.
