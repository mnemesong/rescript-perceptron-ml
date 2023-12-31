// Generated by ReScript, PLEASE EDIT WITH CARE
'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Belt_Array = require("rescript/lib/js/belt_Array.js");
var Belt_Option = require("rescript/lib/js/belt_Option.js");
var Belt_Result = require("rescript/lib/js/belt_Result.js");
var MlPerceptron = require("../src/MlPerceptron.bs.js");
var Caml_exceptions = require("rescript/lib/js/caml_exceptions.js");
var MlPerceptronLayer = require("../src/MlPerceptronLayer.bs.js");
var Caml_js_exceptions = require("rescript/lib/js/caml_js_exceptions.js");
var Mocha$RescriptMocha = require("rescript-mocha/lib/js/src/Mocha.bs.js");
var Assert$RescriptMocha = require("rescript-mocha/lib/js/src/Assert.bs.js");
var MlActivationFunction = require("../src/MlActivationFunction.bs.js");
var MlErrorMetricFunction = require("../src/MlErrorMetricFunction.bs.js");

var L1Count = {
  layerCount: 10
};

var L2Count = {
  layerCount: 4
};

var L3Count = {
  layerCount: 4
};

var L4Count = {
  layerCount: 1
};

var L1 = MlPerceptronLayer.MakePerceptronLayerLimitedArr(L1Count);

var L2 = MlPerceptronLayer.MakePerceptronLayerLimitedArr(L2Count);

var L3 = MlPerceptronLayer.MakePerceptronLayerLimitedArr(L3Count);

var L4 = MlPerceptronLayer.MakePerceptronLayerLimitedArr(L4Count);

var RecognizerPerceptron = MlPerceptron.MakePerceptron4(L1, L2, L3, L4, MlActivationFunction.ActivFuncSigmoid, MlErrorMetricFunction.ErrorMetricEuclidean);

var L3LayerGetResultException = /* @__PURE__ */Caml_exceptions.create("MlPerceptronTestL4.L3LayerGetResultException");

var StudePerceptronGetResultExn = /* @__PURE__ */Caml_exceptions.create("MlPerceptronTestL4.StudePerceptronGetResultExn");

Mocha$RescriptMocha.describe("test 2 and 8 recognizer Perceptron l4")(undefined, undefined, undefined, (function (param) {
        var makeDataSet = (function(cnt) {
    let result = [];
    for(let i = 0; i < cnt; i++) {
      let val = Array(10).fill(0);
      let r = Math.floor(Math.random() * 10);
      if(r === 11) {
        i--;
        continue;
      }
      val[r] = 1;
      result.push({i: val, o: ((r === 2) || (r === 8)) ? 1 : 0});
    }
    return result;
  });
        var weightsInit = Curry._1(RecognizerPerceptron.init, (function (param, param$1, param$2) {
                return Math.random() * 2.0 - 1.0;
              }));
        var dataSet = makeDataSet(100000);
        var tmp;
        var exit = 0;
        var x;
        try {
          x = Belt_Array.reduce(dataSet, weightsInit, (function (acc, el) {
                  var l4layer = Belt_Result.getExn(Curry._1(L4.arrToLayer, [el.o]));
                  try {
                    return Belt_Result.getExn(Curry._4(RecognizerPerceptron.stude, el.i, l4layer, acc, 0.01));
                  }
                  catch (exn){
                    throw {
                          RE_EXN_ID: StudePerceptronGetResultExn,
                          Error: new Error()
                        };
                  }
                }));
          exit = 1;
        }
        catch (raw_e){
          var e = Caml_js_exceptions.internalToOCamlException(raw_e);
          console.log(e);
          tmp = {
            TAG: /* Error */1,
            _0: e
          };
        }
        if (exit === 1) {
          tmp = {
            TAG: /* Ok */0,
            _0: x
          };
        }
        var studedWeights = Belt_Result.getExn(tmp);
        var dataSet2 = makeDataSet(1000);
        var tmp$1;
        var exit$1 = 0;
        var x$1;
        try {
          x$1 = Belt_Array.map(dataSet2, (function (dr) {
                  var sol = Belt_Option.getExn(Belt_Array.get(Curry._1(L4.layerToArr, Belt_Result.getExn(Curry._2(RecognizerPerceptron.solve, dr.i, studedWeights))), 0));
                  return (dr.o - sol) * (dr.o - sol) < 0.05;
                }));
          exit$1 = 1;
        }
        catch (raw_e$1){
          var e$1 = Caml_js_exceptions.internalToOCamlException(raw_e$1);
          console.log(e$1);
          tmp$1 = {
            TAG: /* Error */1,
            _0: e$1
          };
        }
        if (exit$1 === 1) {
          tmp$1 = {
            TAG: /* Ok */0,
            _0: x$1
          };
        }
        var solutions = Belt_Result.getExn(tmp$1);
        var onlyFalsesResults = solutions.filter(function (b) {
              return b === false;
            });
        Mocha$RescriptMocha.it("assert has no fales")(undefined, undefined, undefined, (function (param) {
                Assert$RescriptMocha.ok(onlyFalsesResults.length < 5);
              }));
      }));

exports.L1Count = L1Count;
exports.L2Count = L2Count;
exports.L3Count = L3Count;
exports.L4Count = L4Count;
exports.L1 = L1;
exports.L2 = L2;
exports.L3 = L3;
exports.L4 = L4;
exports.RecognizerPerceptron = RecognizerPerceptron;
exports.L3LayerGetResultException = L3LayerGetResultException;
exports.StudePerceptronGetResultExn = StudePerceptronGetResultExn;
/* L1 Not a pure module */
