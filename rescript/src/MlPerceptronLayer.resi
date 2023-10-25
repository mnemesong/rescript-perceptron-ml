open MlActivationFunction
open MlErrorMetricFunction

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

module MakePerceptronLayerLimitedArr: MakePerceptronLayerLimitedArr

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

module MakePerceptronLayersCross: MakePerceptronLayersCross
