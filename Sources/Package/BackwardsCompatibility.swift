import _Differentiation

@differentiable(wrt: predicted)
public func l1Loss<Scalar: TensorFlowFloatingPoint>(
  predicted: Tensor<Scalar>,
  expected: Tensor<Scalar>
) -> Tensor<Scalar> {
  l1Loss(predicted: predicted, expected: expected, reduction: { $0.sum() })
  // l1Loss(predicted: predicted, expected: expected, reduction: { $0 })
}
