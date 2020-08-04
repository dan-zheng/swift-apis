import _Differentiation

// MARK: Original function

@differentiable
@_silgen_name("foo")
func foo(_ x: Float) -> Float {
  let y = (x * x)
  let z = y + y
  return z
}

// MARK: Manually written pullback structs

struct PullbackAdd {
  func callAsFunction(_ v: Float) -> (dlhs: Float, drhs: Float) {
    return (v, v)
  }
}

struct PullbackMultiply {
  let lhs: Float
  let rhs: Float

  func callAsFunction(_ v: Float) -> (dlhs: Float, drhs: Float) {
    return (v * rhs, v * lhs)
  }
}

extension Float {
  static func vjpAdd(_ lhs: Float, _ rhs: Float) -> (
    value: Float, pullback: PullbackAdd
  ) {
    let value = lhs + rhs
    return (value, PullbackAdd())
  }

  static func vjpMultiply(_ lhs: Float, _ rhs: Float) -> (
    value: Float, pullback: PullbackMultiply
  ) {
    let value = lhs * rhs
    return (value, PullbackMultiply(lhs: lhs, rhs: rhs))
  }
}

struct PullbackFoo {
  let pbMul: PullbackMultiply
  let pbAdd: PullbackAdd

  func callAsFunction(_ v: Float) -> Float {
    let (dy1, dy2) = pbAdd(v)
    let dy = dy1 + dy2
    let (dx1, dx2) = pbMul(dy)
    let dx = dx1 + dx2
    return dx
  }
}

@_silgen_name("vjpFoo")
func vjpFoo(_ x: Float) -> (value: Float, pullback: PullbackFoo) {
  let (y, pbMul) = Float.vjpMultiply(x, x)
  let (z, pbAdd) = Float.vjpAdd(y, y)
  return (z, PullbackFoo(pbMul: pbMul, pbAdd: pbAdd))
}

// Helpers.

var blackHole: Any? = nil

@inline(never)
func consume<T>(_ x: T) {
  blackHole = x
}

@_silgen_name("test_autodiff_gradient_apply")
func test_autodiff_gradient_apply() {
  consume(gradient(of: foo)(10))
}

@_silgen_name("test_manual_gradient_apply")
func test_manual_gradient_apply() {
  consume(vjpFoo(10).pullback(1))
}
