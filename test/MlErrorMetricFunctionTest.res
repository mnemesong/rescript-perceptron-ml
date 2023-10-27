open RescriptMocha
open Mocha
open MlErrorMetricFunction

let approxEq = (x: float, y: float): bool => (x -. y) *. (x -. y) < 0.01

describe("test EuclideanErrorMetric", () => {
  it("test 1", () => {
    let result = ErrorMetricEuclidean.errMetricDerivative(1.0, 0.5)->ErrorMetricEuclidean.errToFloat
    let nominal = -0.5
    Assert.ok(approxEq(result, nominal))
  })
})
