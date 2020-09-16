import Benchmark
import _Differentiation

@differentiable
@_silgen_name("foo")
func foo(_ x: Float) -> Float {
  let y = x * x
  let z = y + y
  return z
}

var blackHole: Any? = nil

@inline(never)
func consume<T>(_ x: T) {
  blackHole = x
}

benchmark("forward call") { state in
  try state.measure {
    consume(foo(10))
  }
}

benchmark("gradient call") { state in
  try state.measure {
    consume(gradient(at: 10, in: foo))
  }
}

Benchmark.main()
