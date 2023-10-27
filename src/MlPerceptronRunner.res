open MlPerceptron
open Belt

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

module MakePerceptronRunner: MakePerceptronRunner = (P: Perceptron) => {
  type input<'a> = P.input<'a>
  type output<'a> = P.output<'a>
  type weights = P.weights

  let checkMass = (
    initor: unit => datarow<input<float>, output<float>>,
    weights: weights,
    count: int,
  ): result<array<datacompars<output<float>, output<float>>>, exn> => {
    let cnt = count > 0 ? Ok(count) : Error(CountShouldBeGreaterThenZero)
    let dataSet = cnt->Result.map(c => Array.range(1, c)->Array.map(_ => initor()))
    Result.flatMap(dataSet, r =>
      Array.reduce(r, (Ok([]): result<array<datacompars<output<float>, output<float>>>, exn>), (
        acc,
        ds,
      ) =>
        acc->Result.flatMap(
          a => {
            let datacompars = P.solve(ds.i, weights)->Result.map(r => {o: r, n: ds.o})
            datacompars->Result.map(dc => a->Array.concat([dc]))
          },
        )
      )
    )
  }

  let studeMass = (
    initor: unit => datarow<input<float>, output<float>>,
    weights: weights,
    studeCoeff: float,
    count: int,
  ): result<weights, exn> => {
    let cnt = count > 0 ? Ok(count) : Error(CountShouldBeGreaterThenZero)
    let dataSet = cnt->Result.map(c => Array.range(1, c)->Array.map(_ => initor()))
    Result.flatMap(dataSet, dataSet =>
      Array.reduce(dataSet, Ok(weights), (acc, ds) => {
        acc->Result.flatMap(w => {P.stude(ds.i, ds.o, w, studeCoeff)})
      })
    )
  }
}
