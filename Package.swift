// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "TF-1302",
  dependencies: [
    .package(name: "Benchmark", url: "https://github.com/google/swift-benchmark", .branch("master"))
  ],
  targets: [
    .target(
      name: "TF-1302",
      dependencies: ["Benchmark"]),
  ]
)
