import _Differentiation

public protocol TensorFlowFloatingPoint: Differentiable & FloatingPoint where Self == TangentVector {}

public struct Tensor<Scalar> {}
extension Tensor: Differentiable where Scalar: Differentiable {}

public extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func sum() -> Tensor { self }
}

public extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(where Scalar: TensorFlowFloatingPoint)
  static func -(lhs: Self, rhs: Self) -> Self { lhs }
}

@inlinable
@differentiable(where T: TensorFlowFloatingPoint)
public func abs<T: SignedNumeric>(_ x: Tensor<T>) -> Tensor<T> {
  // _Raw.abs(x)
  x
}
