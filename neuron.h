#ifndef NEURAL_NET_NEURON_H_
#define NEURAL_NET_NEURON_H_

// A very simple neuron class.

#include <vector>

#include "output_functions.h"

namespace network {

class Neuron {
public:
  Neuron();
  // Sets the impulse function for the neuron. It should be differentiable if
  // back propagation is to be used. Note that the instance of ImpulseFunction
  // should be allocated somewhere where it won't dissapear.
  inline void SetOutputFunction(ImpulseFunction *impulse) {
    impulse_ = impulse;
  }
  // Returns a pointer to the neuron's impulse function.
  inline ImpulseFunction *GetOutputFunction() {
    return impulse_;
  }
  // Set the neuron's bias weight, which defaults to 0.
  inline void SetBias(double bias) {
    bias_ = bias;
  }
  // Sets the neuron's inputs to the contents of a vector.
  inline void SetInputs(const std::vector<double>& values) {
    inputs_ = values;
  }
  // Sets the neuron's input's weights to the contents of a vector.
  inline void SetWeights(const std::vector<double>& values) {
    weights_ = values;
  }
  // Changes the weights according to a back propagated signal.
  bool AdjustWeights(double learning_rate, double signal);
  // Gets the neuron's current weights.
  inline void GetWeights(std::vector<double> *weights) {
    *weights = weights_;
  }
  // Gets the neuron's current inputs.
  inline void GetInputs(std::vector<double> *inputs) {
    *inputs = inputs_;
  }
  // Writes the output of the neuron to output. Returns true upon success and
  // false upon error.
  bool GetOutput(double *output);
  // Return the number of weights currently set.
  inline int GetNumWeights() {
    return weights_.size();
  }
private:
  // The neuron's impulse function.
  ImpulseFunction *impulse_;
  // The bias weight.
  double bias_;
  // The value of the neuron's inputs.
  std::vector<double> inputs_;
  // The value of the weight on each input.
  std::vector<double> weights_;
};

} //network

#endif
