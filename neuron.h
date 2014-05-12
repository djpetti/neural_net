#ifndef NEURAL_NET_NEURON_H_
#define NEURAL_NET_NEURON_H_

// A very simple neuron class.

#include <vector>

#include "output_functions.h"

namespace network {

class Neuron {
public:
  Neuron();
  ~Neuron();
  // Sets the impulse function for the neuron. It should be differentiable if
  // back propagation is to be used. Note that the instance of ImpulseFunction
  // should be allocated somewhere where it won't dissapear.
  inline void SetOutputFunction(ImpulseFunction *impulse) {
    impulse_ = impulse;
    own_impulse_ = false;
  }
  // Returns a pointer to the neuron's impulse function.
  inline ImpulseFunction *GetOutputFunction() {
    return impulse_;
  }
  // Set the neuron's bias weight, which defaults to 0.
  inline void SetBias(double bias) {
    bias_ = bias;
  }
  // Returns the neuron's bias weight.
  inline double GetBias() {
    return bias_;
  }
  // Sets the neuron's inputs to the contents of a vector.
  inline void SetInputs(const std::vector<double>& values) {
    inputs_ = values;
  }
  // Sets the neuron's input's weights to the contents of a vector.
  void SetWeights(const std::vector<double>& values);
  // Changes the weights according to a back propagated signal.
  bool AdjustWeights(double learning_rate, double momentum, double signal);
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
  // The following two functions are used for the back propagation algorithm.
  // Returns weights in from the list in reverse order.
  bool GetLastWeight(double *weight, double *input);
  // The next thing GetLastWeight will return is at the end of the weights list.
  void Reset();

private:
  // The neuron's impulse function.
  ImpulseFunction *impulse_;
  // Whether or not we're responsible for our impulse function.
  bool own_impulse_ = true;
  // The bias weight.
  double bias_;
  // The index of the weight that GetLastWeight will return next.
  int weight_i_;
  // The last output of this neuron.
  double last_output_;
  // The value of the neuron's inputs.
  std::vector<double> inputs_;
  // The value of the weight on each input.
  std::vector<double> weights_;
  // The value of the weights before they were changed by backpropagation.
  std::vector<double> old_weights_;
  // The last change to each of our weights.
  std::vector<double> delta_weights_;
};

} //network

#endif
