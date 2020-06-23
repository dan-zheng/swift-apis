// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import Benchmark

@inline(never)
func blackHole<T>(_ x: T) {}

let AutoDiffBenchmarks = BenchmarkSuite(name: "AutoDiff") { suite in
  suite.benchmark("identity") {
    func f(_ x: Float) -> Float {
      x
    }
    for _ in 0..<100_000 {
      blackHole(valueWithGradient(at: 1, in: f))
    }
  }

  suite.benchmark("square") {
    func f(_ x: Float) -> Float {
      x * x
    }
    for _ in 0..<100_000 {
      blackHole(valueWithGradient(at: 1, in: f))
    }
  }

  let onesArray: [Float] = Array(repeating: 1, count: 50)

  suite.benchmark("array sum (raw loop)") {
    func sum(_ array: [Float]) -> Float {
      var result: Float = 0
      for i in withoutDerivative(at: 0..<array.count) {
        result += array[i]
      }
      return result
    }
    for _ in 0..<100 {
      blackHole(valueWithGradient(at: onesArray, in: sum))
    }
  }

  suite.benchmark("array sum (differentiableReduce)") {
    func sum(_ array: [Float]) -> Float {
      var result: Float = 0
      for i in withoutDerivative(at: 0..<array.count) {
        result += array[i]
      }
      return result
    }
    for _ in 0..<100 {
      blackHole(valueWithGradient(at: onesArray, in: sum))
    }
  }
}
