// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

internal protocol _AnyDifferentiableBox {
  // `Differentiable` requirements.
  mutating func _move(along direction: AnyDerivative)

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: Differentiable>(to type: U.Type) -> U?
}

internal struct _ConcreteDifferentiableBox<T: Differentiable>: _AnyDifferentiableBox
{
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    return _base
  }

  func _unboxed<U: Differentiable>(to type: U.Type) -> U? {
    return (self as? _ConcreteDifferentiableBox<U>)?._base
  }

  mutating func _move(along direction: AnyDerivative) {
    guard
      let directionBase =
        direction.base as? T.TangentVector
    else {
      fatalError()
    }
    _base.move(along: directionBase)
  }
}

public struct AnyDifferentiable: Differentiable {
  internal var _box: _AnyDifferentiableBox

  internal init(_box: _AnyDifferentiableBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T: Differentiable>(_ base: T) {
    self._box = _ConcreteDifferentiableBox<T>(base)
  }

  @inlinable
  @derivative(of: init)
  internal static func _vjpInit<T: Differentiable>(
    _ base: T
  ) -> (value: AnyDifferentiable, pullback: (AnyDerivative) -> T.TangentVector)
  {
    return (AnyDifferentiable(base), { v in v.base as! T.TangentVector })
  }

  @inlinable
  @derivative(of: init)
  internal static func _jvpInit<T: Differentiable>(
    _ base: T
  ) -> (
    value: AnyDifferentiable, differential: (T.TangentVector) -> AnyDerivative
  ) {
    return (AnyDifferentiable(base), { dbase in AnyDerivative(dbase) })
  }

  public typealias TangentVector = AnyDerivative

  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction)
  }
}

internal class _AnyLayerBox<Input: Differentiable, Output: Differentiable> {
  // `Differentiable` requirements.
  func _move(along direction: AnyLayerTangentVector) {
    fatalError()
  }

  // `Layer` requirements.
  func _callAsFunction(_ input: Input) -> Output {
    fatalError()
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    fatalError()
  }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: Layer>(to type: U.Type) -> U? {
    fatalError()
  }
}

internal final class _ConcreteLayerBox<T: Layer> : _AnyLayerBox<T.Input, T.Output> {
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  override var _typeErasedBase: Any {
    return _base
  }

  override func _unboxed<U: Layer>(to type: U.Type) -> U? {
    return (self as? _ConcreteLayerBox<U>)?._base
  }

  // `Differentiable` requirements.
  override func _move(along direction: AnyLayerTangentVector) {
    guard let directionBase = direction.base as? T.TangentVector else {
      // _derivativeTypeMismatch(T.self, type(of: direction._typeErasedBase))
      fatalError()
    }
    _base.move(along: directionBase)
  }

  // `Layer` requirements.
  override func _callAsFunction(_ input: T.Input) -> T.Output {
    return _base.callAsFunction(input)
  }
}

public struct AnyLayer<Input: Differentiable, Output: Differentiable>: Layer {
  internal var _box: _AnyLayerBox<Input, Output>

  internal init(_box: _AnyLayerBox<Input, Output>) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  public init<T: Layer>(_ base: T) where Input == T.Input, Output == T.Output {
    self._box = _ConcreteLayerBox<T>(base)
  }

  // `Differentiable` requirements.
  public typealias TangentVector = AnyLayerTangentVector

  public mutating func move(along direction: TangentVector) {
    _box._move(along: direction)
  }

  // `EuclideanDifferentiable` requirements.
  public var differentiableVectorView: TangentVector {
    fatalError()
  }

  // `Layer` requirements.
  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    return _box._callAsFunction(input)
  }
}

/*
internal protocol _AnyLayerTangentVectorBox {
  // `Differentiable` requirements.
  func _move(along direction: AnyLayerTangentVector)

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U: Layer>(to type: U.Type) -> U?
}

internal class _ConcreteLayerTangentVectorBox : _AnyLayerTangentVectorBox {
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  override var _typeErasedBase: Any {
    return _base
  }

  override func _unboxed<U: Layer>(to type: U.Type) -> U? {
    return (self as? _ConcreteLayerTangentVectorBox<U>)?._base
  }

  override func _move(along direction: AnyLayerTangentVector) {
    guard let directionBase =
      direction.base as? T.TangentVector else {
      // _derivativeTypeMismatch(T.self, type(of: direction._typeErasedBase))
      fatalError()
    }
    _base.move(along: directionBase)
  }
}

struct AnyLayerTangentVector: VectorProtocol & ElementaryFunctions & PointwiseMultiplicative & KeyPathIterable {
  internal var _box: _AnyLayerTangentVectorBox

  internal init(_box: _AnyLayerTangentVectorBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  // @differentiable(jvp: _jvpInit(_:), vjp: _vjpInit(_:))
  public init<T: Layer>(_ base: T) where Input == T.Input, Output == T.Output {
    self._box = _ConcreteLayerBox<T>(base)
  }

  typealias TangentVector = Self
}
*/

internal protocol _AnyLayerTangentVectorBox {
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  func _isEqual(to other: _AnyLayerTangentVectorBox) -> Bool
  func _isNotEqual(to other: _AnyLayerTangentVectorBox) -> Bool

  // `AdditiveArithmetic` requirements.
  static var _zero: _AnyLayerTangentVectorBox { get }
  func _adding(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox
  func _subtracting(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox

  // `Differentiable` requirements.
  mutating func _move(along direction: _AnyLayerTangentVectorBox)

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: _AnyLayerTangentVectorBox { get }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any { get }

  /// Returns the underlying value unboxed to the given type, if possible.
  func _unboxed<U>(to type: U.Type) -> U?
    where U : Differentiable, U.TangentVector == U
}

extension _AnyLayerTangentVectorBox {
  /// Returns true if the underlying value has type `AnyLayerTangentVector.OpaqueZero`.
  func _isOpaqueZero() -> Bool {
    return _unboxed(to: AnyLayerTangentVector.OpaqueZero.self) != nil
  }
}

@inline(never)
@usableFromInline
internal func _derivativeTypeMismatch(
  _ x: Any.Type, _ y: Any.Type, file: StaticString = #file, line: UInt = #line
) -> Never {
  preconditionFailure("""
    Derivative type mismatch: \
    \(String(reflecting: x)) and \(String(reflecting: y))
    """, file: file, line: line)
}

internal struct _ConcreteAnyLayerTangentVectorBox<T> : _AnyLayerTangentVectorBox
  where T : Differentiable, T.TangentVector == T
{
  /// The underlying base value.
  var _base: T

  init(_ base: T) {
    self._base = base
  }

  /// The underlying base value, type-erased to `Any`.
  var _typeErasedBase: Any {
    return _base
  }

  func _unboxed<U>(to type: U.Type) -> U?
    where U : Differentiable, U.TangentVector == U
  {
    return (self as? _ConcreteAnyLayerTangentVectorBox<U>)?._base
  }

  // `Equatable` requirements (implied by `AdditiveArithmetic`).

  func _isEqual(to other: _AnyLayerTangentVectorBox) -> Bool {
    return _base == other._unboxed(to: T.self)
  }

  func _isNotEqual(to other: _AnyLayerTangentVectorBox) -> Bool {
    return _base != other._unboxed(to: T.self)
  }

  // `AdditiveArithmetic` requirements.

  static var _zero: _AnyLayerTangentVectorBox {
    return _ConcreteAnyLayerTangentVectorBox(T.zero)
  }

  func _adding(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
    // 0 + x = x
    if _isOpaqueZero() {
      return x
    }
    // y + 0 = y
    if x._isOpaqueZero() {
      return self
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteAnyLayerTangentVectorBox(_base + xBase)
  }

  func _subtracting(_ x: _AnyLayerTangentVectorBox) -> _AnyLayerTangentVectorBox {
    // y - 0 = y
    if x._isOpaqueZero() {
      return self
    }
    // 0 - x = -x
    if _isOpaqueZero() {
      return type(of: x)._zero._subtracting(x)
    }
    guard let xBase = x._unboxed(to: T.self) else {
      _derivativeTypeMismatch(T.self, type(of: x._typeErasedBase))
    }
    return _ConcreteAnyLayerTangentVectorBox(_base - xBase)
  }

  // `Differentiable` requirements.

  mutating func _move(along direction: _AnyLayerTangentVectorBox) {
    if direction._isOpaqueZero() {
      return
    }
    // The case where `self._isOpaqueZero()` returns true is handled in
    // `AnyLayerTangentVector.move(along:)`.
    guard let directionBase =
      direction._unboxed(to: T.TangentVector.self) else {
      _derivativeTypeMismatch(T.self, type(of: direction._typeErasedBase))
    }
    _base.move(along: directionBase)
  }

  // `EuclideanDifferentiable` requirements.
  var _differentiableVectorView: _AnyLayerTangentVectorBox {
    return self
  }
}

/// A type-erased derivative value.
///
/// The `AnyLayerTangentVector` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable` and
/// `AdditiveArithmetic`, hiding the specifics of the underlying value.
// public struct AnyLayerTangentVector : EuclideanDifferentiable & AdditiveArithmetic {
public struct AnyLayerTangentVector: VectorProtocol & ElementaryFunctions & PointwiseMultiplicative & KeyPathIterable & EuclideanDifferentiable & AdditiveArithmetic {
  internal var _box: _AnyLayerTangentVectorBox

  internal init(_box: _AnyLayerTangentVectorBox) {
    self._box = _box
  }

  /// The underlying base value.
  public var base: Any {
    return _box._typeErasedBase
  }

  /// Creates a type-erased derivative from the given derivative.
  @differentiable
  public init<T>(_ base: T) where T : Differentiable, T.TangentVector == T {
    self._box = _ConcreteAnyLayerTangentVectorBox<T>(base)
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _vjpInit<T>(
    _ base: T
  ) -> (value: AnyLayerTangentVector, pullback: (AnyLayerTangentVector) -> T.TangentVector)
    where T : Differentiable, T.TangentVector == T
  {
    return (AnyLayerTangentVector(base), { v in v.base as! T.TangentVector })
  }

  @derivative(of: init)
  @usableFromInline
  internal static func _jvpInit<T>(
    _ base: T
  ) -> (value: AnyLayerTangentVector, differential: (T.TangentVector) -> AnyLayerTangentVector)
    where T : Differentiable, T.TangentVector == T
  {
    return (AnyLayerTangentVector(base), { dbase in AnyLayerTangentVector(dbase) })
  }

  public typealias TangentVector = AnyLayerTangentVector

  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  public static func == (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs._box._isEqual(to: rhs._box)
  }
  public static func != (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) -> Bool {
    return lhs._box._isNotEqual(to: rhs._box)
  }

  // `AdditiveArithmetic` requirements.

  /// Internal struct representing an opaque zero value.
  @frozen
  @usableFromInline
  internal struct OpaqueZero : EuclideanDifferentiable & AdditiveArithmetic {}

  public static var zero: AnyLayerTangentVector {
    return AnyLayerTangentVector(
      _box: _ConcreteAnyLayerTangentVectorBox<OpaqueZero>(OpaqueZero.zero))
  }

  public static func + (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> AnyLayerTangentVector {
    return AnyLayerTangentVector(_box: lhs._box._adding(rhs._box))
  }

  @derivative(of: +)
  @usableFromInline internal static func _vjpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) -> (AnyLayerTangentVector, AnyLayerTangentVector)) {
    return (lhs + rhs, { v in (v, v) })
  }

  @derivative(of: +)
  @usableFromInline internal static func _jvpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
    differential: (AnyLayerTangentVector, AnyLayerTangentVector) -> (AnyLayerTangentVector)) {
      return (lhs + rhs, { (dlhs, drhs) in dlhs + drhs })
  }

  public static func - (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> AnyLayerTangentVector {
    return AnyLayerTangentVector(_box: lhs._box._subtracting(rhs._box))
  }

  @derivative(of: -)
  @usableFromInline internal static func _vjpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) -> (AnyLayerTangentVector, AnyLayerTangentVector)) {
    return (lhs - rhs, { v in (v, .zero - v) })
  }

  @derivative(of: -)
  @usableFromInline internal static func _jvpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) -> (value: AnyLayerTangentVector,
        differential: (AnyLayerTangentVector, AnyLayerTangentVector) -> AnyLayerTangentVector) {
    return (lhs - rhs, { (dlhs, drhs) in dlhs - drhs })
  }

  // `VectorProtocol` requirements.
  public static var one: AnyLayerTangentVector {
    fatalError()
/*
    return AnyLayerTangentVector(
      _box: _ConcreteAnyLayerTangentVectorBox<OpaqueZero>(OpaqueZero.zero))
*/
  }

  public var reciprocal: AnyLayerTangentVector {
    fatalError()
/*
    return AnyLayerTangentVector(
      _box: _ConcreteAnyLayerTangentVectorBox<OpaqueZero>(OpaqueZero.zero))
*/
  }

  public static func .* (lhs: Self, rhs: Self) -> Self {
    return AnyLayerTangentVector(_box: lhs._box._subtracting(rhs._box))
  }

  public typealias VectorSpaceScalar = Float

  public func adding(_ x: VectorSpaceScalar) -> Self {
    fatalError()
  }

  public func subtracting(_ x: VectorSpaceScalar) -> Self {
    fatalError()
  }

  public func scaled(by scalar: VectorSpaceScalar) -> Self {
    fatalError()
  }

  // `ElementaryFunctions` requirements.
  public static func sqrt(_ x: Self) -> Self {
    fatalError()
  }
  public static func cos(_ x: Self) -> Self {
    fatalError()
  }
  public static func sin(_ x: Self) -> Self {
    fatalError()
  }
  public static func tan(_ x: Self) -> Self {
    fatalError()
  }
  public static func acos(_ x: Self) -> Self {
    fatalError()
  }
  public static func asin(_ x: Self) -> Self {
    fatalError()
  }
  public static func atan(_ x: Self) -> Self {
    fatalError()
  }
  public static func cosh(_ x: Self) -> Self {
    fatalError()
  }
  public static func sinh(_ x: Self) -> Self {
    fatalError()
  }
  public static func tanh(_ x: Self) -> Self {
    fatalError()
  }
  public static func acosh(_ x: Self) -> Self {
    fatalError()
  }
  public static func asinh(_ x: Self) -> Self {
    fatalError()
  }
  public static func atanh(_ x: Self) -> Self {
    fatalError()
  }
  public static func exp(_ x: Self) -> Self {
    fatalError()
  }
  public static func exp2(_ x: Self) -> Self {
    fatalError()
  }
  public static func exp10(_ x: Self) -> Self {
    fatalError()
  }
  public static func expm1(_ x: Self) -> Self {
    fatalError()
  }
  public static func log(_ x: Self) -> Self {
    fatalError()
  }
  public static func log2(_ x: Self) -> Self {
    fatalError()
  }
  public static func log10(_ x: Self) -> Self {
    fatalError()
  }
  public static func log1p(_ x: Self) -> Self {
    fatalError()
  }
  public static func pow(_ x: Self, _ y: Self) -> Self {
    fatalError()
  }
  public static func pow(_ x: Self, _ n: Int) -> Self {
    fatalError()
  }
  public static func root(_ x: Self, _ n: Int) -> Self {
    fatalError()
  }

  // `Differentiable` requirements.
  public mutating func move(along direction: TangentVector) {
    if _box._isOpaqueZero() {
      _box = direction._box
      return
    }
    _box._move(along: direction._box)
  }

  // `EuclideanDifferentiable` requirements.
  public var differentiableVectorView: TangentVector {
    return self
  }
}
