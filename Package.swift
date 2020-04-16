// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "Package",
  products: [
    .library(
      name: "Package",
      type: .dynamic,
      targets: ["Package"]),
  ],
  dependencies: [],
  targets: [
    .target(
      name: "Package",
      dependencies: []),
    .testTarget(
      name: "PackageTests",
      dependencies: ["Package"]),
  ]
)
