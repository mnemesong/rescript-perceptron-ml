open MlActivationFunction
open MlErrorMetricFunction
open Belt

type activFuncValue = {
  solve: float,
  derivative: float,
}

module type PerceptronLayersCross = {
  type layer1<'a>
  type layer2<'a>
  type weights = layer1<layer2<float>>
  type errVal

  let input: layer1<float> => layer1<activFuncValue>
  let solve: (layer1<activFuncValue>, weights) => layer2<activFuncValue>
  let findError: (layer2<activFuncValue>, layer2<float>) => layer2<errVal>
  let backpropagadeError: (layer1<activFuncValue>, layer2<errVal>, weights) => layer1<errVal>
  let weightCorrection: (layer1<activFuncValue>, layer2<errVal>, weights, float) => weights
  let init: ((int, int) => float) => weights
}

module type PerceptronLayer = {
  type layer<'a>

  let layerToArr: layer<'a> => array<'a>
  let arrToLayer: array<'a> => option<layer<'a>>
  let init: (int => 'a) => layer<'a>
}

module type LayerCount = {
  let layerCount: int
}

module type MakePerceptronLayerLimitedArr = (LayerCount: LayerCount) => PerceptronLayer

module MakePerceptronLayerLimitedArr: MakePerceptronLayerLimitedArr = (LayerCount: LayerCount) => {
  type layer<'a>

  let layerToArr: layer<'a> => array<'a> = %raw(`
  function (vals) {
    return vals;
  }
  `)

  let arrToLayer: array<'a> => option<layer<'a>> = arr => {
    let convert: array<'a> => layer<'a> = %raw(`
    function (vals) {
      return vals;
    }
    `)
    Array.length(arr) === LayerCount.layerCount ? Some(convert(arr)) : None
  }

  let init = (initor: int => 'a): layer<'a> => {
    let il: (int => 'a, int) => layer<'a> = %raw(`
    function (initor, cnt) {
      let result = [];
      for(let i = 0; i < cnt; i++) {
        result.push(initor(i));
      }
      return result;
    }
    `)
    il(initor, LayerCount.layerCount)
  }
}

module type MakePerceptronLayersCross = (
  Layer1: PerceptronLayer,
  Layer2: PerceptronLayer,
  F: ActivFunc,
  Err: ErrorMetric,
) =>
(
  PerceptronLayersCross
    with type layer1<'a> = Layer1.layer<'a>
    and type layer2<'a> = Layer2.layer<'a>
    and type errVal = Err.errVal
)

module MakePerceptronLayersCross: MakePerceptronLayersCross = (
  Layer1: PerceptronLayer,
  Layer2: PerceptronLayer,
  F: ActivFunc,
  Err: ErrorMetric,
) => {
  type layer1<'a> = Layer1.layer<'a>
  type layer2<'a> = Layer2.layer<'a>
  type weights = layer1<layer2<float>>
  type errVal = Err.errVal

  let input = (inp: layer1<float>): layer1<activFuncValue> => {
    let arr1 = Layer1.layerToArr(inp)
    let resArr = arr1->Array.map(v => {
      solve: F.solve(v),
      derivative: F.derivative(v),
    })
    Layer1.arrToLayer(resArr)->Option.getExn
  }

  let solve = (l1: layer1<activFuncValue>, w: weights): layer2<activFuncValue> => {
    let layer1Arr = Layer1.layerToArr(l1)
    let weightsArr = Layer1.layerToArr(w)->Array.map(a => Layer2.layerToArr(a))
    let layer2SumsArr =
      weightsArr[0]
      ->Option.getExn
      ->Array.mapWithIndex((n, _) => {
        let valsWeighted =
          layer1Arr->Array.mapWithIndex((i, v) =>
            v.solve *. (weightsArr[i]->Option.getExn)[n]->Option.getExn
          )
        Array.reduce(valsWeighted, 0.0, (acc, el) => acc +. el)
      })
    let layer2Arr = layer2SumsArr->Array.map(v => {
      solve: F.solve(v),
      derivative: F.derivative(v),
    })
    Layer2.arrToLayer(layer2Arr)->Option.getExn
  }

  let findError = (vals: layer2<activFuncValue>, nominals: layer2<float>): layer2<errVal> => {
    let valsArr = Layer2.layerToArr(vals)
    nominals
    ->Layer2.layerToArr
    ->Array.mapWithIndex((i, n) => Err.errMetricDerivative(n, (valsArr[i]->Option.getExn).solve))
    ->Layer2.arrToLayer
    ->Option.getExn
  }

  let backpropagadeError = (
    l1Vals: layer1<activFuncValue>,
    l2Errs: layer2<errVal>,
    weights: weights,
  ): layer1<errVal> => {
    let weightsArr = Layer1.layerToArr(weights)->Array.map(w => Layer2.layerToArr(w))
    l1Vals
    ->Layer1.layerToArr
    ->Array.mapWithIndex((i, l1v) => {
      let errSum =
        l2Errs
        ->Layer2.layerToArr
        ->Array.mapWithIndex((j, el) => (Err.errToFloat(el), j))
        ->Array.reduce(0.0, (acc, el) => {
          let (errFl, j) = el
          acc +. errFl *. (weightsArr[i]->Option.getExn)[j]->Option.getExn
        })
      Err.floatToErr(errSum *. l1v.derivative)
    })
    ->Layer1.arrToLayer
    ->Option.getExn
  }

  let weightCorrection = (
    l1Vals: layer1<activFuncValue>,
    l2Errs: layer2<errVal>,
    weights: weights,
    learnCoeff: float,
  ): weights => {
    let l1ValsArr = Layer1.layerToArr(l1Vals)
    let l2ErrArr = Layer2.layerToArr(l2Errs)
    let weightsArr = Layer1.layerToArr(weights)->Array.map(w => Layer2.layerToArr(w))
    let newWeightsArr = weightsArr->Array.mapWithIndex((i, wi) =>
      Array.mapWithIndex(wi, (j, wj) => {
        let solve = (l1ValsArr[i]->Option.getExn).solve
        let err = l2ErrArr[j]->Option.getExn->Err.errToFloat
        wj -. learnCoeff *. solve *. err
      })
    )
    newWeightsArr
    ->Array.map(nw => nw->Layer2.arrToLayer->Option.getExn)
    ->Layer1.arrToLayer
    ->Option.getExn
  }

  let init = (initor: (int, int) => float): weights => {
    Layer1.init(i => {
      Layer2.init(j => initor(i, j))
    })
  }
}
