// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "TSanTest",
  platforms: [
    .macOS(.v10_13)
  ],
  dependencies: [],
  targets: [
    .target(
      name: "TSanTest",
      dependencies: []),
    .testTarget(
      name: "TSanTestTests",
      dependencies: ["TSanTest"]),
  ]
)
