module type ActivFunc = {
  let solve: float => float
  let derivative: float => float
}

module ActivFuncSigmoid = {
  let solve = (x: float) => 1.0 /. (1.0 +. Js.Math.exp(-.x))
  let derivative = (x: float) => {
    let s = solve(x)
    s *. (1.0 -. s)
  }
}

module ActivFuncReLU = {
  let solve = (x: float) => Js.Math.max_float(0.0, x)
  let derivative = (x: float) => x > 0.0 ? 1.0 : 0.0
}

module ActivFuncTh = {
  let solve = (x: float) =>
    (Js.Math.exp(x) -. Js.Math.exp(-.x)) /. (Js.Math.exp(x) +. Js.Math.exp(-.x))
  let derivative = (x: float) => {
    let s = solve(x)
    1.0 -. s *. s
  }
}
