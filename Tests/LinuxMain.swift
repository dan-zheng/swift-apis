import TSanTestTests
import XCTest

var tests = [XCTestCaseEntry]()
tests += TSanTestTests.allTests()
XCTMain(tests)
