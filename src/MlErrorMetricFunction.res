module type ErrorMetric = {
  type errVal

  let errToFloat: errVal => float
  let floatToErr: float => errVal
  let errMetricDerivative: (float, float) => errVal
}

module ErrorMetricEuclidean: ErrorMetric = {
  type errVal = float

  let errToFloat = (errVal: errVal): float => errVal
  let floatToErr = (f: float): float => f
  let errMetricDerivative = (nominal: float, result: float): errVal => floatToErr(result -. nominal)
}
