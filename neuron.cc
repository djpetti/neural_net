#include <stdint.h>

#include "neuron.h"

namespace network {

Neuron::Neuron() : 
    impulse_(nullptr), 
    bias_(0),
    weight_i_(-1) {}

void Neuron::SetWeights(const std::vector<double>& values) {
  weights_ = values;

  // delta_weights are now invalid, so reset them to zero.
  delta_weights_.clear();
  for (uint32_t i = 0; i < weights_.size(); ++i) {
    delta_weights_.push_back(0);
  }

  Reset();
}

bool Neuron::AdjustWeights(double learning_rate, double momentum, double error,
    bool print) {
  std::vector<double> weights_buffer;
  if (weights_.size() == inputs_.size()) {
    double signal = impulse_->Derivative(last_output_) * error; 
    // Adjust bias, which is basically a weight with the input permanently set
    // at 1.
    SetBias(bias_ + (learning_rate * signal));

    for (uint32_t i = 0; i < weights_.size(); ++i) {
      double delta = learning_rate * signal * inputs_[i];
      if (print) {
        printf("Input: %f\n", inputs_[i]);
      }
      delta += delta_weights_[i] * momentum;
      weights_[i] += delta;
      if (print) {
        printf("Delta: %f\n", delta);
        printf("Weight: %f\n", weights_[i]);
      }
      weights_buffer.push_back(delta);
    }
    delta_weights_.swap(weights_buffer);
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
      //printf("Inputs: %f, Weights: %f\n", inputs_[i], weights_[i]);
    }
    //printf("Sum: %f\n", sum);
 
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

bool Neuron::GetLastWeight(double *weight, double *input) {
  if (weight_i_ >= 0) {
    *weight = old_weights_[weight_i_];
    *input = inputs_[weight_i_--];
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
