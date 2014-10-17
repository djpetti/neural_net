#include <stdint.h>

#include "neuron.h"

namespace network {

Neuron::Neuron() :
    // The default impulse function is no impulse function, aka. DumbOutputer.
    impulse_(new DumbOutputer()),
    bias_(0),
    weight_i_(-1) {}

Neuron::~Neuron() {
  if (own_impulse_) {
    delete impulse_;
  }
}

void Neuron::SetWeights(const std::vector<double>& values) {
  weights_ = values;

  // delta_weights are now invalid, so reset them to zero.
  delta_weights_.clear();
  for (uint32_t i = 0; i < weights_.size(); ++i) {
    delta_weights_.push_back(0);
  }

  Reset();
}

bool Neuron::AdjustWeights(double learning_rate, double momentum, double error) {
  std::vector<double> weights_buffer;
  if (weights_.size() == inputs_.size()) {
    double signal = impulse_->Derivative(last_output_) * error;
    // Adjust bias, which is basically a weight with the input permanently set
    // at 1.
    SetBias(bias_ + (learning_rate * signal));

    for (uint32_t i = 0; i < weights_.size(); ++i) {
      double delta = learning_rate * signal * inputs_[i];
      delta += delta_weights_[i] * momentum;
      weights_[i] += delta;
      weights_buffer.push_back(delta);
    }
    delta_weights_.swap(weights_buffer);
    return true;
  }

  return false;
}

bool Neuron::GetOutput(double *output) {
  if (inputs_.size() == weights_.size()) {
    // Calculate the initial sum.
    double sum = bias_;
    for (uint32_t i = 0; i < inputs_.size(); ++i) {
      sum += inputs_[i] * weights_[i];
    }

    // Apply the impulse function.
    *output = impulse_->Function(sum);
    last_output_ = *output;

    // Save the current weights.
    old_weights_ = weights_;

    return true;
  } else {
    return false;
  }
}

bool Neuron::GetLastWeight(double *weight) {
  if (weight_i_ >= 0) {
    *weight = old_weights_[weight_i_--];
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
