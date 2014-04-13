#include "output_functions.h"

namespace network {

double ImpulseFunction::Derivative(double input) {
  ASSERT(differentiable_, 
      "Attempt to take derivative of non-differentiable function.");
  
  double x1 = input - 0.0001;
  double x2 = input + 0.0001;
  double y1 = Function(x1);
  double y2 = Function(x2);
  return (y2 - y1) / (x2 - x1);
}

double Threshold::Function(double input) {
  if (input >= threshold_) {
    return 1;
  } else {
    return 0;
  }
}

} //network
