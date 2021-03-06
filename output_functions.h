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
  virtual ~ImpulseFunction() = default;
  virtual double Function(double input) = 0;
  // User can implement a derivative, causes an assertion failure if sombody
  // tries to use is and it isn't overriden. Because we use it for
  // back-propagation, it takes the output to the function.
  virtual double Derivative(double output);

  DISSALOW_COPY_AND_ASSIGN(ImpulseFunction);
};

// Basically sets output to its input.
class DumbOutputer : public ImpulseFunction {
 public:
  inline virtual double Function(double input) {
    return input;
  }
};

// Does a simple threshold.
class Threshold : public ImpulseFunction {
 public:
  explicit Threshold(double threshold) :
      threshold_(threshold) {}
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
  inline virtual double Derivative(double output) {
    return output * (1 - output);
  }
};

// A hyperbolic tangent function, which is another standard one.
class TanH : public ImpulseFunction {
 public:
  inline virtual double Function(double input) {
    return tanh(input);
  }
  inline virtual double Derivative(double output) {
    return pow((1 / cosh(atanh(output))), 2);
  }
};

// A linear output function, often useful for outputs.
class Linear : public ImpulseFunction {
 public:
  explicit Linear(double slope) :
      slope_(slope) {}
  inline virtual double Function(double input) {
    return slope_ * input;
  }
  inline virtual double Derivative(double output) {
    return slope_;
  }

 private:
  double slope_;
};

} //network

#endif
