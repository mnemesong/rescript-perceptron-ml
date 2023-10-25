open MlPerceptronLayer
open MlActivationFunction
open MlErrorMetricFunction
open Belt

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

  let solve = (inp: input<float>, weights: weights): result<output<float>, exn> => {
    let W(w1x2, w2x3) = weights
    let l1Solvs = L1XL2.input(inp)
    let l2Solvs = Result.flatMap(l1Solvs, r => r->L1XL2.solve(w1x2))
    let l3Solvs = Result.flatMap(l2Solvs, r => r->L2XL3.solve(w2x3))
    l3Solvs
    ->Result.map(r => r->L3.layerToArr->Array.map(fv => fv.solve))
    ->Result.flatMap(r => r->L3.arrToLayer)
  }

  let stude = (inp: input<float>, out: output<float>, weights: weights, studeCoeff: float): result<
    weights,
    exn,
  > => {
    let W(w1x2, w2x3) = weights
    let l1Solvs = L1XL2.input(inp)
    let l2Solvs = Result.flatMap(l1Solvs, r => r->L1XL2.solve(w1x2))
    let l3Solvs = Result.flatMap(l2Solvs, r => r->L2XL3.solve(w2x3))
    let err3 = Result.flatMap(l3Solvs, r => r->L2XL3.findError(out))
    let err2 = Result.flatMap(l2Solvs, r =>
      Result.flatMap(err3, e => L2XL3.backpropagadeError(r, e, w2x3))
    )
    let w1x2' = Result.flatMap(l1Solvs, r =>
      Result.flatMap(err2, e => L1XL2.weightCorrection(r, e, w1x2, studeCoeff))
    )
    let w2x3' = Result.flatMap(l2Solvs, r =>
      Result.flatMap(err3, e => L2XL3.weightCorrection(r, e, w2x3, studeCoeff))
    )
    Result.flatMap(w1x2', w1 => Result.flatMap(w2x3', w2 => Ok(W(w1, w2))))
  }

  let init = (initor: (int, int, int) => float): weights => {
    let w1x2 = L1XL2.init((i, j) => initor(i, j, 1))
    let w2x3 = L2XL3.init((i, j) => initor(i, j, 2))
    W(w1x2, w2x3)
  }
}
