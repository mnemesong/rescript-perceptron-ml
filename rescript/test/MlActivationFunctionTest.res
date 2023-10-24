open RescriptMocha
open Mocha
open MlActivationFunction

let approxEq = (x: float, y: float): bool => (x -. y) *. (x -. y) < 0.01

describe("test ActivFuncSigmoid", () => {
  describe("test solve", () => {
    it(
      "test 1",
      () => {
        let result = ActivFuncSigmoid.solve(0.0)
        let nominal = 0.5
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 2",
      () => {
        let result = ActivFuncSigmoid.solve(-6.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 3",
      () => {
        let result = ActivFuncSigmoid.solve(6.0)
        let nominal = 1.0
        Assert.ok(approxEq(result, nominal))
      },
    )
  })

  describe("test derivarive", () => {
    it(
      "test 1",
      () => {
        let result = ActivFuncSigmoid.derivative(0.0)
        let nominal = 0.25
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 2",
      () => {
        let result = ActivFuncSigmoid.derivative(-6.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 3",
      () => {
        let result = ActivFuncSigmoid.derivative(6.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
  })
})

describe("test ActivFuncReLU", () => {
  describe("test solve", () => {
    it(
      "test 1",
      () => {
        let result = ActivFuncReLU.solve(0.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 2",
      () => {
        let result = ActivFuncReLU.solve(-6.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 3",
      () => {
        let result = ActivFuncReLU.solve(6.0)
        let nominal = 6.0
        Assert.ok(approxEq(result, nominal))
      },
    )
  })

  describe("test derivative", () => {
    it(
      "test 1",
      () => {
        let result = ActivFuncReLU.derivative(0.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 2",
      () => {
        let result = ActivFuncReLU.derivative(-6.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 3",
      () => {
        let result = ActivFuncReLU.derivative(6.0)
        let nominal = 1.0
        Assert.ok(approxEq(result, nominal))
      },
    )
  })
})

describe("test ActivFuncTh", () => {
  describe("test solve", () => {
    it(
      "test 1",
      () => {
        let result = ActivFuncTh.solve(0.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 2",
      () => {
        let result = ActivFuncTh.solve(-6.0)
        let nominal = -1.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 3",
      () => {
        let result = ActivFuncTh.solve(6.0)
        let nominal = 1.0
        Assert.ok(approxEq(result, nominal))
      },
    )
  })

  describe("test derivative", () => {
    it(
      "test 1",
      () => {
        let result = ActivFuncTh.derivative(0.0)
        let nominal = 1.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 2",
      () => {
        let result = ActivFuncTh.derivative(-6.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
    it(
      "test 3",
      () => {
        let result = ActivFuncTh.derivative(6.0)
        let nominal = 0.0
        Assert.ok(approxEq(result, nominal))
      },
    )
  })
})
