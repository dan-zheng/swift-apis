// swift-tools-version:5.3

import PackageDescription

let package = Package(
  name: "AutoDiffBenchmark",
  products: [
    .executable(
      name: "AutoDiffBenchmark",
      targets: ["AutoDiffBenchmark"])
  ],
  dependencies: [
    .package(name: "Benchmark", url: "https://github.com/google/swift-benchmark", .branch("master"))
  ],
  targets: [
    .target(
      name: "AutoDiffBenchmark",
      dependencies: ["Benchmark"],
      swiftSettings: [SwiftSetting.unsafeFlags(["-cross-module-optimization", "-O"])])
  ]
)
