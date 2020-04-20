// LayerWrapper --------------------------------------

public protocol LayerWrapper: Layer {
  @noDerivative var isAnnotated: Bool { get set }

  @differentiable
  func forward(_ input: Input) -> Output

  associatedtype Input
  associatedtype Output
}

public extension LayerWrapper {
  @differentiable
  func callAsFunction(_ input: Input) -> Output {
    let activation = forward(input)
    return activation
  }
}

public extension LayerWrapper where Input == Tensor<Float>, Output == Tensor<Float> {
  @differentiable
  func annotated(_ input: Input) -> Output {
    #if USING_X10_BACKEND
      let annotation = "type=\(type(of: self))"
      let annotated = input.annotate(annotation)
      return annotated
    #else
      return input
    #endif
  }

  @differentiable
  func callAsFunction(_ input: Input) -> Output {
    let activation = forward(input)

    // Store layer statistics
    if isAnnotated {
      return annotated(activation)
    } else {
      return activation
    }
  }

  /// Returns the annotations obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: All collected annotations from the XLA graph.
  func annotations(input: Input) -> String {
    return self.annotations(inputShape: input.shape)
  }

  /// Returns the annotations obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The shape of the input to the layer.
  /// - Returns: All collected annotations from the XLA graph.
  func annotations(inputShape: TensorShape) -> String {
    #if USING_X10_BACKEND
      LazyTensorBarrier()
      let zeros = Tensor<Float>(repeating: 0, shape: inputShape, on: Device.defaultXLA)
      let model = type(of: self).init(copying: self, to: Device.defaultXLA)
    #else
      let zeros = Tensor<Float>(repeating: 0, shape: inputShape)
      let model = self
    #endif
    let output = model.forward(zeros)

    return output.annotations
  }

  func summary(input: Input) -> String {
    return self.annotations(input: input)
  }

  func summary(inputShape: TensorShape) -> String {
    return self.annotations(inputShape: inputShape)
  }
}


// Models --------------------------------------

public struct SummaryNetWrapped: LayerWrapper {
  @noDerivative public var isAnnotated: Bool
  public var dense1: Dense<Float>
  public var dense2: Dense<Float>
  public var dense3: Dense<Float>

  public init() {
    isAnnotated = false
    dense1 = Dense<Float>(inputSize: 1, outputSize: 1)
    dense2 = Dense<Float>(inputSize: 4, outputSize: 4)
    dense3 = Dense<Float>(inputSize: 4, outputSize: 4)
  }

  @differentiable
  public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
    let layer1 = dense1(input)
    let layer2 = layer1.reshaped(to: [1, 4])
    let layer3 = dense2(layer2)
    let layer4 = dense3(layer3)
    return layer4
  }
}

public struct SummaryNetBare: Layer {
  public var dense1: Dense<Float>
  public var dense2: Dense<Float>
  public var dense3: Dense<Float>

  public init() {
    dense1 = Dense<Float>(inputSize: 1, outputSize: 1)
    dense2 = Dense<Float>(inputSize: 4, outputSize: 4)
    dense3 = Dense<Float>(inputSize: 4, outputSize: 4)
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let layer1 = dense1(input)
    let layer2 = layer1.reshaped(to: [1, 4])
    let layer3 = dense2(layer2)
    let layer4 = dense3(layer3)
    return layer4
  }
}

/*
// Layer --------------------------------------
extension Layer where Input == Tensor<Float>, Output == Tensor<Float> {
  /// Returns the annotations obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: All collected annotations from the XLA graph.
  public func annotations(input: Input) -> String {
    return self.annotations(inputShape: input.shape)
  }

  public func summary(input: Input) -> String {
    return self.annotations(input: input)
  }

  /// Returns the annotations obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The shape of the input to the layer.
  /// - Returns: All collected annotations from the XLA graph.
  public func annotations(inputShape: TensorShape) -> String {
    LazyTensorBarrier()
    let zeros = Tensor<Float>(repeating: 0, shape: inputShape, on: Device.defaultXLA)
    let model = type(of: self).init(copying: self, to: Device.defaultXLA)
    let output = model(zeros)

    return output.annotations
  }

  public func summary(inputShape: TensorShape) -> String {
    return self.annotations(inputShape: inputShape)
  }
}

@frozen
public struct DenseWrapped<Scalar: TensorFlowFloatingPoint>: LayerWrapper {
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>
  /// The weight matrix.
  public var weight: Tensor<Scalar>
  /// The bias vector.
  public var bias: Tensor<Scalar>
  /// The element-wise activation function.
  @noDerivative public let activation: Activation
  /// Indicates whether this is a batched dense layer.
  @noDerivative internal let batched: Bool
  /// Workaround optionals not being handled by AD
  @noDerivative private let useBias: Bool
  @noDerivative public var isAnnotated: Bool

  /// The element-wise activation function type.
  public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
//  @noDerivative public var delegates: [(Output) -> ()] = []

  /// Creates an instance from the given weight, optional bias, and activation function.
  ///
  /// - Note: currently, `weight` is the only differentiability parameter. `bias` can be made a
  ///   differentiability parameter after `Optional` conditionally conforms to `Differentiable`:
  ///   TF-499.
  @differentiable(wrt: weight)
  public init(
    weight: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation
  ) {
    precondition(weight.rank <= 3, "The rank of the 'weight' tensor must be less than 4.")
    precondition(
      bias == nil || bias!.rank <= 2, "The rank of the 'bias' tensor must be less than 3.")
    self.weight = weight
    self.bias = bias ?? .zero
    self.activation = activation
    self.batched = weight.rank == 3
    useBias = (bias != nil)
    isAnnotated = true
  }

  // TODO(TF-433): Remove custom derivative after `try_apply` differentiation is supported.
  @derivative(of: init)
  @usableFromInline
  static func vjpInit(
    weight: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation
  ) -> (value: Self, pullback: (TangentVector) -> Tensor<Scalar>) {
    let value = DenseWrapped(weight: weight, bias: bias, activation: activation)
    return (value, { v in v.weight })
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    if batched {
      let hidden = matmul(input.expandingShape(at: 1), weight).squeezingShape(at: 1)
      return activation(useBias ? hidden + bias : hidden)
    }

    return activation(useBias ? (matmul(input, weight) + bias) : matmul(input, weight))
  }
}

extension DenseWrapped {
  /// Creates a `Dense` layer with the specified input size, output size, and element-wise
  /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
  /// the bias vector is created with shape `[outputSize]`.
  ///
  /// - Parameters:
  ///   - inputSize: The dimensionality of the input space.
  ///   - outputSize: The dimensionality of the output space.
  ///   - activation: The activation function to use. The default value is `identity(_:)`.
  ///   - weightInitializer: Initializer to use for `weight`.
  ///   - biasInitializer: Initializer to use for `bias`.
  public init(
    inputSize: Int,
    outputSize: Int,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    self.init(
      weight: weightInitializer([inputSize, outputSize]),
      bias: useBias ? biasInitializer([outputSize]) : nil,
      activation: activation)
  }
}
*/
