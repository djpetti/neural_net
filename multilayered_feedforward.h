#ifndef NEURAL_NETWORK_MULTILAYERED_FEEDFORWARD_H_
#define NEURAL_NETWORK_MULTILAYERED_FEEDFORWARD_H_

#include <map>
#include <vector>

#include <stdint.h>
#include <string.h>

#include "network.h"
#include "neuron.h"
#include "output_functions.h"

// Contains the necessary code for representing a multilayed-feedforward neural
// network.

namespace network {

class MFNetwork : public Network {
 public:
  // inputs is the number of input neurons, outputs is the number of output
  // neurons, and layer_size is the number of neurons in each hidden layer.
  MFNetwork(int inputs, int outputs, int layer_size);
  virtual ~MFNetwork();
  // Adds a new hidden layer.
  inline void AddHiddenLayer() {
    DoLayerAdd(layer_size_);
  }
  // Returns the number of hidden layers.
  inline int HiddenLayerQuantity() {
    return layers_.size() - 2; // subtract input and output.
  }
  // Writes contents of array values to the inputs. Values must be the same
  // size as the number of inputs.
  void SetInputs(double *values);
  // Computes each neuron for the given inputs and writes the contents to the
  // array values. Values must be able to accomodate a number of items that is
  // at least the number of outputs. Returns true for success, false for
  // failure.
  inline bool GetOutputs(double *values) {
    return DoGetOutputs(values, nullptr);
  }
  // Sets neuron to point to the neuron in the layer specified by layer_i, which
  // indexes from 0, starting with the input layer, and neuron_i, which indexes
  // from 0. Returns nullptr upon failure.
  Neuron *GetNeuron(uint32_t layer_i, uint32_t neuron_i);
  // The following function sets all the neuron's weights to random values. Upper
  // and lower are the inclusive bounds of these values.
  inline void RandomWeights(int lower, int upper) {
    use_random_ = true;
    upper_ = upper;
    lower_ = lower;
  }
  // Sets all the weights in the network to <values>.
  void SetWeights(const std::vector<double>& values);
  // Sets the weights on all the inputs going into <layer_i> to <values>.
  bool SetLayerWeights(uint32_t layer_i, const std::vector<double>& values);
  // Sets the same impulse function for all the neurons.
  void SetOutputFunctions(ImpulseFunction *impulse);
  // Sets the same impulse function for all the neurons in a layer. <layer_i> is
  // the index of said layer.
  bool SetLayerOutputFunctions(uint32_t layer_i, 
      ImpulseFunction *impulse);
  // Specifies that a neuron in a layer ouputs to a specific group of neurons in
  // the next layer. All neurons and layers are referred to by index. Returns
  // true for success, false if any of the indices given are invalid.
  bool SetOutputRoute(uint32_t layer_i, uint32_t neuron_i, 
    const std::vector<int>& output_nodes);
  // Copies the architechture of <source> into this network, but keeps the weights of
  // this network set to 1.
  bool CopyLayout(const MFNetwork& source);
  // Allows the user to specify the learning rate coefficient for the
  // back-propagation algorithm. (The default is 0.1.)
  inline void SetLearningRate(const double rate) {
    learning_rate_ = rate;
  }
  // Propagates an error through the network, adjusting weights as it goes.
  // Although it can return false, the only time it should really do so is if
  // you're trying to propagate an error through a network which can't give you
  // a valid output in the first place.
  bool PropagateError(const double error);
  // Constructs a network with the exact same architechture as this one. It
  // allocates it on the heap, and the caller MUST take ownership of it.
  // TODO(danielp): Get rid of this method, it's unnecessary and dumb.
  virtual bool Clone(MFNetwork *dest);
  // Copies a specific neuron routing layout from one hidden layer to another
  // one. Returns true for success, false if the indices are invalid.
  //bool CopyOutputRoute(int source_layer_i, int dest_layer_i);
  // This one is for genetic algorithms. It returns the total number of weights.
  virtual size_t GetChromosomeSize();
  // Returns an array of the weights of the neurons. chromosome must have enough space
  // to store at least the number returned by GetChromosomeSize.
  virtual bool GetChromosome(uint64_t *chromosome);
  // Sets all the weights in the network to those specified by the array
  // chromosome. Note that weights must have a size equal to the number
  // GetNumWeights returns.
  virtual bool SetChromosome(uint64_t *chromosome);
  // Saves a network to a file for future use. Returns true if it writes
  // successfully, false if it doesn't.
  static bool SaveToFile(const char *path, MFNetwork *network);
  // Reads a network previously saved to a file into memory.
  static bool ReadFromFile(const char *path, MFNetwork *dest);

 private:
  // A struct to represent a layer.
  struct Layer_t {
    // Note that the MFNetwork destructor is responsible for freeing these
    // pointers.
    std::vector<Neuron *> Neurons;
    // Specifies which neurons in the next layer the output from each neuron in
    // this layer is routed to.
    std::map<int, std::vector<int> > RoutingMap; 
  };

  // Layer adding helper, neurons is the number of neurons in the new layer.
  void DoLayerAdd(int neurons);
  // Helper for getting the output, that can be customized depending on what
  // type of learning the network is doing.
  bool DoGetOutputs(double *values, std::vector<double> *osubj);

  uint32_t num_inputs_;
  uint32_t num_outputs_;
  uint32_t layer_size_;
  // Upper and lower bounds for random weights.
  int upper_;
  int lower_;
  // Back-propagation learning rate.
  double learning_rate_;
  // Whether random weights were requested.
  bool use_random_;
  // A vector of all our hidden layers. The MFNetwork destructor is also
  // responsible for freeing these.
  std::vector<Layer_t *> layers_;
  // A map used to temporarily store input for each neuron in a layer.
  std::map<int, std::vector<double> > layer_input_buffer_;
};

} //network

#endif
