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

// MARK: KeyPathIterable

/// An implementation detail of `KeyPathIterable`; do not use this protocol
/// directly.
public protocol _KeyPathIterableBase {
  var _allKeyPathsTypeErased: [AnyKeyPath] { get }
  var _recursivelyAllKeyPathsTypeErased: [AnyKeyPath] { get }
}

/// A type whose values provides custom key paths to properties or elements.
public protocol KeyPathIterable: _KeyPathIterableBase {
  /// A type that can represent a collection of all key paths of this type.
  associatedtype AllKeyPaths: Collection
    where AllKeyPaths.Element == PartialKeyPath<Self>

  /// A collection of all custom key paths of this value.
  var allKeyPaths: AllKeyPaths { get }
}

public extension KeyPathIterable {
  /// An array of all custom key paths of this value and any custom key paths
  /// nested within each of what this value's key paths refers to.
  var recursivelyAllKeyPaths: [PartialKeyPath<Self>] {
    var result: [PartialKeyPath<Self>] = []
    for kp in allKeyPaths {
      result.append(kp)
      if let nested = self[keyPath: kp] as? _KeyPathIterableBase {
        for nkp in nested._recursivelyAllKeyPathsTypeErased {
          result.append(kp.appending(path: nkp)!)
        }
      }
    }
    return result
  }
}

public extension KeyPathIterable {
  var _allKeyPathsTypeErased: [AnyKeyPath] {
    return allKeyPaths.map { $0 as AnyKeyPath }
  }
  var _recursivelyAllKeyPathsTypeErased: [AnyKeyPath] {
    return recursivelyAllKeyPaths.map { $0 as AnyKeyPath }
  }
}

public extension KeyPathIterable {
  /// Returns an array of all custom key paths of this value, to the specified
  /// type.
  func allKeyPaths<T>(to _: T.Type) -> [KeyPath<Self, T>] {
    return allKeyPaths.compactMap { $0 as? KeyPath<Self, T> }
  }

  /// Returns an array of all custom key paths of this value and any custom key
  /// paths nested within each of what this value's key paths refers to, to
  /// the specified type.
  func recursivelyAllKeyPaths<T>(to _: T.Type) -> [KeyPath<Self, T>] {
    return recursivelyAllKeyPaths.compactMap { $0 as? KeyPath<Self, T> }
  }

  /// Returns an array of all custom writable key paths of this value, to the
  /// specified type.
  func allWritableKeyPaths<T>(to _: T.Type) -> [WritableKeyPath<Self, T>] {
    return allKeyPaths(to: T.self)
      .compactMap { $0 as? WritableKeyPath<Self, T> }
  }

  /// Returns an array of all custom writable key paths of this value and any
  /// custom writable key paths nested within each of what this value's key
  /// paths refers to, to the specified type.
  func recursivelyAllWritableKeyPaths<T>(
    to _: T.Type
  ) -> [WritableKeyPath<Self, T>] {
    return recursivelyAllKeyPaths(to: T.self)
      .compactMap { $0 as? WritableKeyPath<Self, T> }
  }
}

// MARK: Collection conformances

/// Returns `true` if all of the given key paths are instances of
/// `WritableKeyPath<Root, Value>`.
private func areWritable<Root, Value>(
  _ keyPaths: [PartialKeyPath<Root>], valueType: Value.Type
) -> Bool {
  return !keyPaths.contains(
    where: { kp in !(kp is WritableKeyPath<Root, Value>) }
  )
}

extension Array: KeyPathIterable {
  public typealias AllKeyPaths = [PartialKeyPath<Array>]
  public var allKeyPaths: [PartialKeyPath<Array>] {
    let result = indices.map { \Array[$0] }
    _internalInvariant(areWritable(result, valueType: Element.self))
    return result
  }
}

// TODO(TF-938): Remove this conformance after removing
// `Element: Differentiable` requirement.
//
// Currently necessary to avoid error:
//
//   error: conditional conformance of type 'Array<Element>.DifferentiableView'
//   to protocol 'KeyPathIterable' does not imply conformance to inherited
//   protocol '_KeyPathIterableBase'.
extension Array.DifferentiableView: _KeyPathIterableBase
where Element: Differentiable {}

// TODO(TF-938): Remove `Element: Differentiable` requirement.
extension Array.DifferentiableView: KeyPathIterable
where Element: Differentiable {
  public typealias AllKeyPaths = [PartialKeyPath<Array.DifferentiableView>]
  public var allKeyPaths: [PartialKeyPath<Array.DifferentiableView>] {
    let result = [\Array.DifferentiableView.base]
    _internalInvariant(areWritable(result, valueType: Array.self))
    return result
  }
}

extension Dictionary: KeyPathIterable {
  public typealias AllKeyPaths = [PartialKeyPath<Dictionary>]
  public var allKeyPaths: [PartialKeyPath<Dictionary>] {
    // Note: `Dictionary.subscript(_: Key)` returns `Value?` and can be used to
    // form `WritableKeyPath<Self, Value>` key paths.
    // Force-unwrapping the result is necessary.
    let result = keys.map { \Dictionary[$0]! }
    _internalInvariant(areWritable(result, valueType: Value.self))
    return result
  }
}

