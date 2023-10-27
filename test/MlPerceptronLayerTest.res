open RescriptMocha
open Mocha
open MlPerceptronLayer
open MlActivationFunction
open MlErrorMetricFunction
open Belt

let approxEq = (x: float, y: float): bool => (x -. y) *. (x -. y) < 0.01

module type Layer1 = {
  include PerceptronLayer
  let construct: ('a, 'a, 'a) => layer<'a>
  let deconstruct: layer<'a> => ('a, 'a, 'a)
  let init: (int => 'a) => layer<'a>
}

module Layer1: Layer1 = {
  type layer<'a>

  let construct: ('a, 'a, 'a) => layer<'a> = %raw(`
  function (a, b, c) {
    return [a, b, c];
  }
  `)

  let deconstruct: layer<'a> => ('a, 'a, 'a) = %raw(`
  function (vals) {
    return vals;
  }
  `)

  let layerToArr: layer<'a> => array<'a> = %raw(`
  function (a) {
    return a;
  }
  `)

  let arrToLayer: array<'a> => result<layer<'a>, exn> = x => {
    let f = %raw(`
    function (a) {
      return a;
    }
    `)
    Ok(f(x))
  }

  let init: (int => 'a) => layer<'a> = %raw(`
  function (f) {
    return [f(0), f(1), f(2)];
  }
  `)
}

module type Layer2 = {
  include PerceptronLayer
  let construct: ('a, 'a) => layer<'a>
  let deconstruct: layer<'a> => ('a, 'a)
  let init: (int => 'a) => layer<'a>
}

module Layer2: Layer2 = {
  type layer<'a>

  let construct: ('a, 'a) => layer<'a> = %raw(`
  function (a1, a2) {
    return [a1, a2];
  }
  `)

  let deconstruct: layer<'a> => ('a, 'a) = %raw(`
  function (vals) {
    return vals;
  }
  `)

  let layerToArr: layer<'a> => array<'a> = %raw(`
  function (a) {
    return a;
  }
  `)

  let arrToLayer: array<'a> => result<layer<'a>, exn> = x => {
    let f = %raw(`
    function (a) {
      return a;
    }
    `)
    Ok(f(x))
  }

  let init: (int => 'a) => layer<'a> = %raw(`
  function (f) {
    return [f(0), f(1)];
  }
  `)
}

module PerceptronLayer = MakePerceptronLayersCross(
  Layer1,
  Layer2,
  ActivFuncSigmoid,
  ErrorMetricEuclidean,
)

describe("test perceptron", () => {
  describe("test input", () => {
    let given = Layer1.construct(0.67, 3.78, -1.56)
    let (
      {solve: rs1, derivative: rd1},
      {solve: rs2, derivative: rd2},
      {solve: rs3, derivative: rd3},
    ) =
      PerceptronLayer.input(given)->Result.getExn->Layer1.deconstruct
    it("rs1 eq", () => {Assert.ok(approxEq(rs1, 0.66))})
    it("rs2 eq", () => {Assert.ok(approxEq(rs2, 0.97))})
    it("rs3 eq", () => {Assert.ok(approxEq(rs3, 0.17))})
    it("rd1 eq", () => {Assert.ok(approxEq(rd1, 0.22))})
    it("rd2 eq", () => {Assert.ok(approxEq(rd2, 0.02))})
    it("rd3 eq", () => {Assert.ok(approxEq(rd3, 0.14))})
  })

  describe("test solve", () => {
    let givenL1Vals = Layer1.construct(
      {solve: 0.66, derivative: 0.22},
      {solve: 0.97, derivative: 0.02},
      {solve: 0.17, derivative: 0.14},
    )
    let givenWeights = Layer1.construct(
      Layer2.construct(0.76, 0.25),
      Layer2.construct(0.15, -1.67),
      Layer2.construct(-5.12, 3.11),
    )
    let ({solve: rs1, derivative: rd1}, {solve: rs2, derivative: rd2}) =
      PerceptronLayer.solve(givenL1Vals, givenWeights)->Result.getExn->Layer2.deconstruct
    it("rs1 eq", () => {Assert.ok(approxEq(rs1, 0.44))})
    it("rs2 eq", () => {Assert.ok(approxEq(rs2, 0.28))})
    it("rd1 eq", () => {Assert.ok(approxEq(rd1, 0.25))})
    it("rd2 eq", () => {Assert.ok(approxEq(rd2, 0.20))})
  })

  describe("test findError", () => {
    let givenL2Vals = Layer2.construct(
      {solve: 0.44, derivative: 0.25},
      {solve: 0.28, derivative: 0.20},
    )
    let givenL2Nominals = Layer2.construct(0.46, 0.5)
    let (ev1, ev2) =
      PerceptronLayer.findError(givenL2Vals, givenL2Nominals)->Result.getExn->Layer2.deconstruct
    it("ev1 eq", () => {Assert.ok(approxEq(ev1->ErrorMetricEuclidean.errToFloat, -0.02))})
    it("ev2 eq", () => {Assert.ok(approxEq(ev2->ErrorMetricEuclidean.errToFloat, -0.22))})
  })

  describe("test backpropagadeError", () => {
    let givenL1Vals = Layer1.construct(
      {solve: 0.66, derivative: 0.22},
      {solve: 0.97, derivative: 0.02},
      {solve: 0.17, derivative: 0.14},
    )
    let givenWeights = Layer1.construct(
      Layer2.construct(0.76, 0.25),
      Layer2.construct(0.15, -1.67),
      Layer2.construct(-5.12, 3.11),
    )
    let givenError = Layer2.construct(
      ErrorMetricEuclidean.floatToErr(0.87),
      ErrorMetricEuclidean.floatToErr(-0.1),
    )
    let (err1, err2, err3) =
      PerceptronLayer.backpropagadeError(givenL1Vals, givenError, givenWeights)
      ->Result.getExn
      ->Layer1.deconstruct
    it("err1 eq", () => {Assert.ok(approxEq(err1->ErrorMetricEuclidean.errToFloat, 0.14))})
    it("err2 eq", () => {Assert.ok(approxEq(err2->ErrorMetricEuclidean.errToFloat, 0.0))})
    it("err3 eq", () => {Assert.ok(approxEq(err3->ErrorMetricEuclidean.errToFloat, -0.66))})
  })

  describe("test weightCorrection", () => {
    let givenL1Vals = Layer1.construct(
      {solve: 0.66, derivative: 0.22},
      {solve: 0.97, derivative: 0.02},
      {solve: 0.17, derivative: 0.14},
    )
    let givenWeights = Layer1.construct(
      Layer2.construct(0.76, 0.25),
      Layer2.construct(0.15, -1.67),
      Layer2.construct(-5.12, 3.11),
    )
    let givenError = Layer2.construct(
      ErrorMetricEuclidean.floatToErr(0.87),
      ErrorMetricEuclidean.floatToErr(-0.1),
    )
    let (w1, w2, w3) =
      PerceptronLayer.weightCorrection(givenL1Vals, givenError, givenWeights, 0.1)
      ->Result.getExn
      ->Layer1.deconstruct
    let (w11, w12) = w1->Layer2.deconstruct
    let (w21, w22) = w2->Layer2.deconstruct
    let (w31, w32) = w3->Layer2.deconstruct
    it("w11 eq", () => {Assert.ok(approxEq(w11, 0.7))})
    it("w12 eq", () => {Assert.ok(approxEq(w12, 0.25))})
    it("w21 eq", () => {Assert.ok(approxEq(w21, 0.07))})
    it("w22 eq", () => {Assert.ok(approxEq(w22, -1.66))})
    it("w31 eq", () => {Assert.ok(approxEq(w31, -5.13))})
    it("w32 eq", () => {Assert.ok(approxEq(w32, 3.11))})
  })

  describe("test init", () => {
    let (w1, w2, w3) =
      PerceptronLayer.init((_, _) => Js.Math.random() *. 2.0 -. 1.0)->Layer1.deconstruct
    let (w11, w12) = w1->Layer2.deconstruct
    let (w21, w22) = w2->Layer2.deconstruct
    let (w31, w32) = w3->Layer2.deconstruct
    let allVals = [w11, w12, w21, w22, w31, w32]
    let zeroVals = Array.reduce(
      allVals,
      [],
      (acc: array<float>, el) => el == 0.0 ? acc->Array.concat([el]) : acc,
    )
    let uniqVals = Array.reduce(
      allVals,
      [],
      (acc, el) => Js.Array2.includes(acc, el) ? acc : acc->Array.concat([el]),
    )
    it("no zeros", () => {Assert.equal(zeroVals->Array.length, 0)})
    it("all uniqs", () => {Assert.equal(uniqVals->Array.length, allVals->Array.length)})
  })
})
