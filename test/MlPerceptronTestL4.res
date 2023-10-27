open RescriptMocha
open Mocha
open MlErrorMetricFunction
open MlActivationFunction
open MlPerceptronLayer
open MlPerceptron
open Belt

module L1Count: LayerCount = {
  let layerCount = 10
}
module L2Count: LayerCount = {
  let layerCount = 4
}
module L3Count: LayerCount = {
  let layerCount = 4
}
module L4Count: LayerCount = {
  let layerCount = 1
}

module L1 = MakePerceptronLayerLimitedArr(L1Count)
module L2 = MakePerceptronLayerLimitedArr(L2Count)
module L3 = MakePerceptronLayerLimitedArr(L3Count)
module L4 = MakePerceptronLayerLimitedArr(L4Count)

module RecognizerPerceptron = MakePerceptron4(
  L1,
  L2,
  L3,
  L4,
  ActivFuncSigmoid,
  ErrorMetricEuclidean,
)

type datarow<'i, 'o> = {i: 'i, o: 'o}

exception L3LayerGetResultException
exception StudePerceptronGetResultExn

describe("test 2 and 8 recognizer Perceptron l4", () => {
  let makeDataSet: int => array<datarow<L1.layer<float>, float>> = %raw(`
  function(cnt) {
    let result = [];
    for(let i = 0; i < cnt; i++) {
      let val = Array(10).fill(0);
      let r = Math.floor(Math.random() * 10);
      if(r === 11) {
        i--;
        continue;
      }
      val[r] = 1;
      result.push({i: val, o: ((r === 2) || (r === 8)) ? 1 : 0});
    }
    return result;
  }
  `)
  let weightsInit = RecognizerPerceptron.init((_, _, _) => Js.Math.random() *. 2.0 -. 1.0)
  let dataSet = makeDataSet(100000)
  let studedWeights = switch Array.reduce(dataSet, weightsInit, (acc, el) => {
    let {i: inp, o: out} = el
    let l4layer = L4.arrToLayer([out])->Result.getExn
    try {RecognizerPerceptron.stude(inp, l4layer, acc, 0.01)->Result.getExn} catch {
    | _ => raise(StudePerceptronGetResultExn)
    }
  }) {
  | x => Ok(x)
  | exception e => {
      Js.Console.log(e)
      Error(e)
    }
  }->Result.getExn
  let dataSet2 = makeDataSet(1000)
  let solutions = switch Array.map(dataSet2, dr => {
    let sol =
      (
        RecognizerPerceptron.solve(dr.i, studedWeights)->Result.getExn->L4.layerToArr
      )[0]->Option.getExn
    (dr.o -. sol) *. (dr.o -. sol) < 0.05
  }) {
  | x => Ok(x)
  | exception e => {
      Js.Console.log(e)
      Error(e)
    }
  }->Result.getExn
  let onlyFalsesResults = Js.Array2.filter(solutions, b => b == false)
  it("assert has no fales", () => {Assert.ok(Array.length(onlyFalsesResults) < 5)})
})
