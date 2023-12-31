// Generated by ReScript, PLEASE EDIT WITH CARE
'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Js_math = require("rescript/lib/js/js_math.js");
var Belt_Array = require("rescript/lib/js/belt_Array.js");
var Belt_Option = require("rescript/lib/js/belt_Option.js");
var Belt_Result = require("rescript/lib/js/belt_Result.js");
var MlPerceptron = require("../src/MlPerceptron.bs.js");
var MlPerceptronLayer = require("../src/MlPerceptronLayer.bs.js");
var MlPerceptronRunner = require("../src/MlPerceptronRunner.bs.js");
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
  layerCount: 1
};

var L1 = MlPerceptronLayer.MakePerceptronLayerLimitedArr(L1Count);

var L2 = MlPerceptronLayer.MakePerceptronLayerLimitedArr(L2Count);

var L3 = MlPerceptronLayer.MakePerceptronLayerLimitedArr(L3Count);

var RecognizerPerceptron = MlPerceptron.MakePerceptron3(L1, L2, L3, MlActivationFunction.ActivFuncSigmoid, MlErrorMetricFunction.ErrorMetricEuclidean);

var RecognizerPerceptronRunner = MlPerceptronRunner.MakePerceptronRunner(RecognizerPerceptron);

Mocha$RescriptMocha.describe("test 3 and 7 recognizer PerceptronRunner l3")(undefined, undefined, undefined, (function (param) {
        var weightsStart = Curry._1(RecognizerPerceptron.init, (function (param, param$1, param$2) {
                return Math.random() * 3.0 - 1.5;
              }));
        var datarowBuilder = function (param) {
          var emptyInp = Belt_Array.range(0, 9);
          var dig = Js_math.floor_int(Math.random() * 10.0);
          var inp = Belt_Result.getExn(Curry._1(L1.arrToLayer, Belt_Array.mapWithIndex(emptyInp, (function (i, param) {
                          if (i === dig) {
                            return 1.0;
                          } else {
                            return 0.0;
                          }
                        }))));
          return {
                  i: inp,
                  o: Belt_Result.getExn(Curry._1(L3.arrToLayer, [dig === 3 || dig === 7 ? 1.0 : 0.0]))
                };
        };
        var weightsFinal = Belt_Result.getExn(Curry._4(RecognizerPerceptronRunner.studeMass, datarowBuilder, weightsStart, 0.01, 100000));
        var result = Curry._3(RecognizerPerceptronRunner.checkMass, datarowBuilder, weightsFinal, 1000);
        var getLayer3Val = function (x) {
          return Belt_Option.getExn(Belt_Array.get(Curry._1(L3.layerToArr, x), 0));
        };
        var onlyFalsy = Belt_Result.getExn(result).filter(function (dc) {
              var diff = getLayer3Val(dc.o) - getLayer3Val(dc.n);
              return diff * diff > 0.05;
            });
        Mocha$RescriptMocha.it("falsy values lesser then 1%")(undefined, undefined, undefined, (function (param) {
                if (onlyFalsy.length > 10) {
                  console.log("Faly values count: ", onlyFalsy.length);
                  Belt_Array.forEach([
                        0,
                        2,
                        4,
                        6
                      ], (function (i) {
                          console.log("Result example: ", Belt_Array.get(Belt_Result.getExn(result), i));
                        }));
                }
                Assert$RescriptMocha.ok(onlyFalsy.length <= 10);
              }));
      }));

exports.L1Count = L1Count;
exports.L2Count = L2Count;
exports.L3Count = L3Count;
exports.L1 = L1;
exports.L2 = L2;
exports.L3 = L3;
exports.RecognizerPerceptron = RecognizerPerceptron;
exports.RecognizerPerceptronRunner = RecognizerPerceptronRunner;
/* L1 Not a pure module */
