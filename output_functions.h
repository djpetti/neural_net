#ifndef NEURAL_NET_OUTPUT_FUNCTIONS_H_
#define NEURAL_NET_OUTPUT_FUNCTIONS_H_

// A compilation of commonly used output functions for neurons, as well as the
// tools to write a custom one.

#include <math.h>

#include "macros.h"

namespace network {

// A basic superclass for impulse functions.
class ImpulseFunction {
 public:
  ImpulseFunction() = default;
  virtual double Function(double input) = 0;
  // User can implement a derivative, falls back on linear approximation if a
  // derivative isn't implemented.
  virtual double Derivative(double input);

  DISSALOW_COPY_AND_ASSIGN(ImpulseFunction);

 protected:
  // A flag the user can unset if their function isn't differentiable accross
  // its domain. It basically causes an assertion failure if it's not set and
  // someone tries to run the Derivative function.
  bool differentiable_ = true;
};

// Basically sets output to its input.
class DumbOutputer : public ImpulseFunction {
 public:
  DumbOutputer() {
    differentiable_ = false;
  }
  inline virtual double Function(double input) {
    return input;
  }
};

// Does a simple threshold.
class Threshold : public ImpulseFunction {
 public:
  explicit Threshold(double threshold) :
      threshold_(threshold) {
    differentiable_ = false;
  }
  virtual double Function(double input);
 
 private:
  double threshold_;
};

// A sigmoid function, which is rather useful.
class Sigmoid : public ImpulseFunction {
 public:
  inline virtual double Function(double input) {
    return 1 / (1 + exp(-input));
  }
  inline virtual double Derivative(double input) {
    double value = Function(input);
    return value * (1 - value);
  }
};

// A hyperbolic tangent function, which is another standard one.
class TanH : public ImpulseFunction {
 public:
  inline virtual double Function(double input) {
    return tanh(input);
  }
  inline virtual double Derivative(double input) {
    return pow((1 / cosh(input)), 2);
  }
};

} //network

#endif
