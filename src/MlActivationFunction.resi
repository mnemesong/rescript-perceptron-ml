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
