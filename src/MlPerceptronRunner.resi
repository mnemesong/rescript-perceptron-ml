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
