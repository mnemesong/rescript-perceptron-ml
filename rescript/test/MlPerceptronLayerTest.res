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
  let arrToLayer: array<'a> => option<layer<'a>> = %raw(`
  function (a) {
    return a;
  }
  `)
}

module type Layer2 = {
  include PerceptronLayer
  let construct: ('a, 'a) => layer<'a>
  let deconstruct: layer<'a> => ('a, 'a)
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
  let arrToLayer: array<'a> => option<layer<'a>> = %raw(`
  function (a) {
    return a;
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
      PerceptronLayer.input(given)->Layer1.deconstruct
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
      PerceptronLayer.solve(givenL1Vals, givenWeights)->Layer2.deconstruct
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
    let (ev1, ev2) = PerceptronLayer.findError(givenL2Vals, givenL2Nominals)->Layer2.deconstruct
    it("ev1 eq", () => {Assert.ok(approxEq(ev1->ErrorMetricEuclidean.errToFloat, -0.02))})
    it("ev2 eq", () => {Assert.ok(approxEq(ev2->ErrorMetricEuclidean.errToFloat, -0.22))})
  })
})
