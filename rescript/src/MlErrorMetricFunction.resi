module type ErrorMetric = {
  type errVal

  let errToFloat: errVal => float
  let floatToErr: float => errVal
  let errMetricDerivative: (float, float) => errVal
}

module ErrorMetricEuclidean: ErrorMetric
