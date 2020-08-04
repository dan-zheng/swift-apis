import Benchmark

// Benchmarking.

benchmark("forward pass") { state in
  try state.measure {
    consume(foo(10))
  }
}

benchmark("gradient (autodiff)") { state in
  try state.measure {
    consume(gradient(of: foo)(10))
  }
}

benchmark("gradient (manual pullback structs)") { state in
  try state.measure {
    consume(vjpFoo(10).pullback(1))
  }
}

Benchmark.main()
