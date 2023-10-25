open MlPerceptronLayer
open MlActivationFunction
open MlErrorMetricFunction
open Belt

module type Perceptron = {
  type input<'a>
  type output<'a>
  type weights

  let solve: (input<float>, weights) => output<float>
  let stude: (input<float>, output<float>, weights, float) => weights
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

module MakePerceptron3: MakePerceptron3 = (
  L1: PerceptronLayer,
  L2: PerceptronLayer,
  L3: PerceptronLayer,
  AF: ActivFunc,
  EM: ErrorMetric,
) => {
  module L1XL2 = MakePerceptronLayersCross(L1, L2, AF, EM)
  module L2XL3 = MakePerceptronLayersCross(L2, L3, AF, EM)

  type input<'a> = L1.layer<'a>
  type output<'a> = L3.layer<'a>
  type weights = w2<L1.layer<L2.layer<float>>, L2.layer<L3.layer<float>>>

  let solve = (inp: input<float>, weights: weights): output<float> => {
    let W(w1x2, w2x3) = weights
    let l1Solvs = L1XL2.input(inp)
    let l2Solvs = L1XL2.solve(l1Solvs, w1x2)
    let l3Solvs = L2XL3.solve(l2Solvs, w2x3)
    l3Solvs->L3.layerToArr->Array.map(fv => fv.solve)->L3.arrToLayer->Option.getExn
  }

  let stude = (
    inp: input<float>,
    out: output<float>,
    weights: weights,
    studeCoeff: float,
  ): weights => {
    let W(w1x2, w2x3) = weights
    let l1Solvs = L1XL2.input(inp)
    let l2Solvs = L1XL2.solve(l1Solvs, w1x2)
    let l3Solvs = L2XL3.solve(l2Solvs, w2x3)
    let err3 = L2XL3.findError(l3Solvs, out)
    let err2 = L2XL3.backpropagadeError(l2Solvs, err3, w2x3)
    let w1x2' = L1XL2.weightCorrection(l1Solvs, err2, w1x2, studeCoeff)
    let w2x3' = L2XL3.weightCorrection(l2Solvs, err3, w2x3, studeCoeff)
    W(w1x2', w2x3')
  }

  let init = (initor: (int, int, int) => float): weights => {
    let w1x2 = L1XL2.init((i, j) => initor(i, j, 1))
    let w2x3 = L2XL3.init((i, j) => initor(i, j, 2))
    W(w1x2, w2x3)
  }
}
