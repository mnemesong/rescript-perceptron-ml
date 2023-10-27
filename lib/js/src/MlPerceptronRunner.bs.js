// Generated by ReScript, PLEASE EDIT WITH CARE
'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Belt_Array = require("rescript/lib/js/belt_Array.js");
var Belt_Result = require("rescript/lib/js/belt_Result.js");
var Caml_exceptions = require("rescript/lib/js/caml_exceptions.js");

var CountShouldBeGreaterThenZero = /* @__PURE__ */Caml_exceptions.create("MlPerceptronRunner.CountShouldBeGreaterThenZero");

function MakePerceptronRunner(P) {
  var checkMass = function (initor, weights, count) {
    var cnt = count > 0 ? ({
          TAG: /* Ok */0,
          _0: count
        }) : ({
          TAG: /* Error */1,
          _0: {
            RE_EXN_ID: CountShouldBeGreaterThenZero
          }
        });
    var dataSet = Belt_Result.map(cnt, (function (c) {
            return Belt_Array.map(Belt_Array.range(1, c), (function (param) {
                          return Curry._1(initor, undefined);
                        }));
          }));
    return Belt_Result.flatMap(dataSet, (function (r) {
                  return Belt_Array.reduce(r, {
                              TAG: /* Ok */0,
                              _0: []
                            }, (function (acc, ds) {
                                return Belt_Result.flatMap(acc, (function (a) {
                                              var datacompars = Belt_Result.map(Curry._2(P.solve, ds.i, weights), (function (r) {
                                                      return {
                                                              o: r,
                                                              n: ds.o
                                                            };
                                                    }));
                                              return Belt_Result.map(datacompars, (function (dc) {
                                                            return Belt_Array.concat(a, [dc]);
                                                          }));
                                            }));
                              }));
                }));
  };
  var studeMass = function (initor, weights, studeCoeff, count) {
    var cnt = count > 0 ? ({
          TAG: /* Ok */0,
          _0: count
        }) : ({
          TAG: /* Error */1,
          _0: {
            RE_EXN_ID: CountShouldBeGreaterThenZero
          }
        });
    var dataSet = Belt_Result.map(cnt, (function (c) {
            return Belt_Array.map(Belt_Array.range(1, c), (function (param) {
                          return Curry._1(initor, undefined);
                        }));
          }));
    return Belt_Result.flatMap(dataSet, (function (dataSet) {
                  return Belt_Array.reduce(dataSet, {
                              TAG: /* Ok */0,
                              _0: weights
                            }, (function (acc, ds) {
                                return Belt_Result.flatMap(acc, (function (w) {
                                              return Curry._4(P.stude, ds.i, ds.o, w, studeCoeff);
                                            }));
                              }));
                }));
  };
  return {
          checkMass: checkMass,
          studeMass: studeMass
        };
}

exports.CountShouldBeGreaterThenZero = CountShouldBeGreaterThenZero;
exports.MakePerceptronRunner = MakePerceptronRunner;
/* No side effect */