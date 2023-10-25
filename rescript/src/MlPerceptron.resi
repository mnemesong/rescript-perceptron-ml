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
