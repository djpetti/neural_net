#include <stdint.h>

#include "neuron.h"

namespace network {

Neuron::Neuron() : 
    impulse_(nullptr), 
    bias_(0),
    weight_i_(-1) {}

bool Neuron::AdjustWeights(double learning_rate, double signal) {
  if (weights_.size() == inputs_.size()) {
    for (uint32_t i = 0; i < weights_.size(); ++i) {
      double delta = learning_rate * signal * inputs_[i];
      weights_[i] += delta;
    }
    return true;
  }

  return false;
}

bool Neuron::GetOutput(double *output) {
  if (inputs_.size() == weights_.size() && impulse_ != nullptr) {
    // Calculate the initial sum.
    double sum = bias_;
    for (uint32_t i = 0; i < inputs_.size(); ++i) {
      sum += inputs_[i] * weights_[i];
    }
 
    // Apply the output function.
    *output = impulse_->Function(sum);
    return true;
  } else {
    return false;
  }
}

bool Neuron::GetLastWeight(double *weight) {
  if (weight_i_ >= 0) {
    *weight = weights_[weight_i_--];
    return true;
  }
  return false;
}

void Neuron::Reset() {
  if (weights_.empty()) {
    weight_i_ = -1;
  } else {
    weight_i_ = weights_.size() - 1;
  }
}

} //network
