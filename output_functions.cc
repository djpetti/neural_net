#include "logger.h"
#include "output_functions.h"

namespace network {

double ImpulseFunction::Derivative(double output) {
  CHECK(false, 
      "Attempt to take derivative of non-differentiable function.");
  return 0;
}

double Threshold::Function(double input) {
  if (input >= threshold_) {
    return 1;
  } else {
    return 0;
  }
}

} //network
