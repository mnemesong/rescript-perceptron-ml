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

  let input: layer1<float> => result<layer1<activFuncValue>, exn>
  let solve: (layer1<activFuncValue>, weights) => result<layer2<activFuncValue>, exn>
  let findError: (layer2<activFuncValue>, layer2<float>) => result<layer2<errVal>, exn>
  let backpropagadeError: (
    layer1<activFuncValue>,
    layer2<errVal>,
    weights,
  ) => result<layer1<errVal>, exn>
  let weightCorrection: (
    layer1<activFuncValue>,
    layer2<errVal>,
    weights,
    float,
  ) => result<weights, exn>
  let init: ((int, int) => float) => weights
}

module type PerceptronLayer = {
  type layer<'a>

  let layerToArr: layer<'a> => array<'a>
  let arrToLayer: array<'a> => result<layer<'a>, exn>
  let init: (int => 'a) => layer<'a>
}

module type LayerCount = {
  let layerCount: int
}

module type MakePerceptronLayerLimitedArr = (LayerCount: LayerCount) => PerceptronLayer

exception InvalidCountOfLayerElements

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

exception WeightsLayerHasNoZeroElement

module MakePerceptronLayersCross: MakePerceptronLayersCross
