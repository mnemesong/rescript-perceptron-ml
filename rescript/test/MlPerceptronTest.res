//open RescriptMocha
//open Mocha
//open MlErrorMetricFunction
//open MlActivationFunction
//open MlPerceptronLayer
//open MlPerceptron
//open Belt
//
//module L1Count: LayerCount = {
//  let layerCount = 10
//}
//module L2Count: LayerCount = {
//  let layerCount = 4
//}
//module L3Count: LayerCount = {
//  let layerCount = 1
//}
//module L1 = MakePerceptronLayerLimitedArr(L1Count)
//module L2 = MakePerceptronLayerLimitedArr(L2Count)
//module L3 = MakePerceptronLayerLimitedArr(L3Count)
//
//module RecognizerPerceptron = MakePerceptron3(L1, L2, L3, ActivFuncSigmoid, ErrorMetricEuclidean)
//
//type datarow<'i, 'o> = {i: 'i, o: 'o}
//
//describe("test 2 and 8 recognizer Perceptron", () => {
//  let makeDataSet: int => array<datarow<L1.layer<float>, float>> = %raw(`
//  function(cnt) {
//    let result = [];
//    for(let i = 0; i < cnt; i++) {
//      let val = Array(10).fill(false);
//      let r = Math.floor(Math.random() * 11);
//      if(r === 11) {
//        i--;
//        continue;
//      }
//      val[r] = true;
//      result.push({i: val, o: ((r === 2) || (r === 8)) ? 1 : 0});
//    }
//    return result;
//  }
//  `)
//  let weightsInit = RecognizerPerceptron.init((_, _, _) => Js.Math.random() *. 2.0 -. 1.0)
//  let dataSet = makeDataSet(10000)
//  let resultWeights = Array.reduce(dataSet, weightsInit, (acc, el) =>
//    try {
//      let {i: inp, o: out} = el
//      RecognizerPerceptron.stude(inp, L3.arrToLayer([out])->Option.getExn, acc, 0.01)
//    } catch {
//    | Js.Exn.Error(e) => {
//        Js.Console.log(e)
//        weightsInit
//      }
//    | _ => {
//        Js.Console.log("Unknown error")
//        weightsInit
//      }
//    }
//  )
//  //Js.Console.log(resultWeights)
//})

