# rescript-perceptron-ml
Perceptron realization on rescript


## Example of usage
```rescript
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

//It's layers built from array of k-th count with assertion - easiest way to create a layer
module L1 = MakePerceptronLayerLimitedArr(L1Count)
module L2 = MakePerceptronLayerLimitedArr(L2Count)
module L3 = MakePerceptronLayerLimitedArr(L3Count)

module RecognizerPerceptron = MakePerceptron3(L1, L2, L3, ActivFuncSigmoid, ErrorMetricEuclidean)
module RecognizerPerceptronRunner = MakePerceptronRunner(RecognizerPerceptron)

//This 3-layers perceptron will start with random weights it will get on input array kind
//of [0, ... 0, 1, 0, ... 0] where [n]th elements is 1, all other is 0.
//It will studed to recognize arrays with 3th and 7th elem is = 1
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
let weightsFinal = //correct recognizing arrays perceptron's weights
  RecognizerPerceptronRunner.studeMass(datarowBuilder, weightsStart, 0.01, 100000)->Result.getExn
//Test that asserts same perceptron studes correctly placed in module MlPerceptronRunnerTest
```


## API

#### MlActivationFunction.resi
Modules provides activation functions for neurons
```rescript
module type ActivFunc = {
  let solve: float => float
  let derivative: float => float
}

//Sigmoid AF
module ActivFuncSigmoid: ActivFunc

//ReLU AF
module ActivFuncReLU: ActivFunc

//Hyperbolic tangens AF
module ActivFuncTh: ActivFunc
```

#### MlErrorMetricFunction
Module provides error-metric functions for perceptron's output
```rescript
module type ErrorMetric = {
  type errVal //its a float, but essense into special type for user not confused it with input or solution

  let errToFloat: errVal => float
  let floatToErr: float => errVal
  let errMetricDerivative: (float, float) => errVal
}

module ErrorMetricEuclidean: ErrorMetric
```

#### MlPerceptronLayer.resi
module provides perceptron layers and layers-cross descriptions
```rescript
open MlActivationFunction
open MlErrorMetricFunction

type activFuncValue = {
  solve: float,
  derivative: float,
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
```

#### MlPerceptron.resi
Module provides perceptron interface and 3-layers and 4-layers ready perceptron realizations
```rescript
open MlPerceptronLayer
open MlActivationFunction
open MlErrorMetricFunction

module type Perceptron = {
  type input<'a>
  type output<'a>
  type weights

  let solve: (input<float>, weights) => result<output<float>, exn>
  let stude: (input<float>, output<float>, weights, float) => result<weights, exn>
  let init: ((int, int, int) => float) => weights
}

type w2<'a, 'b> = W('a, 'b)

module type MakePerceptron3 = (
  L1: PerceptronLayer,
  L2: PerceptronLayer,
  L3: PerceptronLayer,
  AF: ActivFunc,
  EM: ErrorMetric,
) =>
(
  Perceptron
    with type input<'a> = L1.layer<'a>
    and type output<'a> = L3.layer<'a>
    and type weights = w2<L1.layer<L2.layer<float>>, L2.layer<L3.layer<float>>>
)

module MakePerceptron3: MakePerceptron3

type w3<'a, 'b, 'c> = W('a, 'b, 'c)

module type MakePerceptron4 = (
  L1: PerceptronLayer,
  L2: PerceptronLayer,
  L3: PerceptronLayer,
  L4: PerceptronLayer,
  AF: ActivFunc,
  EM: ErrorMetric,
) =>
(
  Perceptron
    with type input<'a> = L1.layer<'a>
    and type output<'a> = L4.layer<'a>
    and type weights = w3<
      L1.layer<L2.layer<float>>,
      L2.layer<L3.layer<float>>,
      L3.layer<L4.layer<float>>,
    >
)

module MakePerceptron4: MakePerceptron4
```

#### MlPerceptronRunner.resi
Runs bult stude of checking for perceptron
```rescript
open MlPerceptron

type datarow<'i, 'o> = {i: 'i, o: 'o}
type datacompars<'o, 'n> = {o: 'o, n: 'n}

module type PerceptronRunner = {
  type input<'a>
  type output<'a>
  type weights

  let checkMass: (
    unit => datarow<input<float>, output<float>>,
    weights,
    int,
  ) => result<array<datacompars<output<float>, output<float>>>, exn>

  let studeMass: (
    unit => datarow<input<float>, output<float>>,
    weights,
    float,
    int,
  ) => result<weights, exn>
}

module type MakePerceptronRunner = (P: Perceptron) =>
(
  PerceptronRunner
    with type input<'a> = P.input<'a>
    and type output<'a> = P.output<'a>
    and type weights = P.weights
)

exception CountShouldBeGreaterThenZero

module MakePerceptronRunner: MakePerceptronRunner
```


## Author
Anatoly Starodubtsev
tostar74@mail.ru


## License
MIT