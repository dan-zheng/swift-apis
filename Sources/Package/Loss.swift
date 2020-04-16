import _Differentiation

@differentiable(wrt: predicted)
public func l1Loss<Scalar: TensorFlowFloatingPoint>(
  predicted: Tensor<Scalar>,
  expected: Tensor<Scalar>,
  reduction: @differentiable (Tensor<Scalar>) -> Tensor<Scalar> = _sum
) -> Tensor<Scalar> {
  reduction(abs(expected - predicted))
}

@differentiable
public func _sum<Scalar: TensorFlowFloatingPoint>(
  _ value: Tensor<Scalar>
) -> Tensor<Scalar> {
  return value.sum()
}
