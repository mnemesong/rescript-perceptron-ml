open RescriptMocha
open Mocha
open MlErrorMetricFunction
open MlActivationFunction
open MlPerceptronLayer
open MlPerceptron
open MlPerceptronRunner
open Belt

module L1Count: LayerCount = {
  let layerCount = 10
}
module L2Count: LayerCount = {
  let layerCount = 4
}
module L3Count: LayerCount = {
  let layerCount = 1
}

module L1 = MakePerceptronLayerLimitedArr(L1Count)
module L2 = MakePerceptronLayerLimitedArr(L2Count)
module L3 = MakePerceptronLayerLimitedArr(L3Count)

module RecognizerPerceptron = MakePerceptron3(L1, L2, L3, ActivFuncSigmoid, ErrorMetricEuclidean)
module RecognizerPerceptronRunner = MakePerceptronRunner(RecognizerPerceptron)

describe("test 3 and 7 recognizer PerceptronRunner l3", () => {
  let weightsStart = RecognizerPerceptron.init((_, _, _) => Js.Math.random() *. 3.0 -. 1.5)
  let datarowBuilder = () => {
    let emptyInp = Array.range(0, 9)
    let dig = Js.Math.floor_int(Js.Math.random() *. 10.0)
    let inp =
      emptyInp->Array.mapWithIndex((i, _) => i == dig ? 1.0 : 0.0)->L1.arrToLayer->Result.getExn
    {
      i: inp,
      o: [dig == 3 || dig == 7 ? 1.0 : 0.0]->L3.arrToLayer->Result.getExn,
    }
  }
  let weightsFinal =
    RecognizerPerceptronRunner.studeMass(datarowBuilder, weightsStart, 0.01, 100000)->Result.getExn
  let result = RecognizerPerceptronRunner.checkMass(datarowBuilder, weightsFinal, 1000)
  let getLayer3Val = (x: RecognizerPerceptron.output<float>): float =>
    (x->L3.layerToArr)[0]->Option.getExn
  let onlyFalsy =
    result
    ->Result.getExn
    ->Js.Array2.filter(dc => {
      let diff = dc.o->getLayer3Val -. dc.n->getLayer3Val
      diff *. diff > 0.05
    })
  it("falsy values lesser then 1%", () => {
    if Array.length(onlyFalsy) > 10 {
      Js.Console.log2("Faly values count: ", Array.length(onlyFalsy))
      [0, 2, 4, 6]->Array.forEach(
        i => Js.Console.log2("Result example: ", (result->Result.getExn)[i]),
      )
    } else {
      ()
    }
    Assert.ok(Array.length(onlyFalsy) <= 10)
  })
})
