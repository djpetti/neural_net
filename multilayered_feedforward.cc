#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "macros.h"
#include "multilayered_feedforward.h"

namespace network {

static_assert(sizeof(double) == sizeof(uint64_t),
            "Otherwise weird things happen.");

MFNetwork::MFNetwork(int inputs, int outputs, int layer_size) :
    num_inputs_(inputs),
    num_outputs_(outputs),
    layer_size_(layer_size),
    learning_rate_(0.1),
    use_random_(false) {
  // Seed random number generator.
  srand(time(NULL));
  // Create the input and output layers.
  DoLayerAdd(num_inputs_);
  DoLayerAdd(num_outputs_);
}

MFNetwork::~MFNetwork() {
  // Free all our heap stuff.
  for (Layer_t *layer : layers_) {
    for (Neuron *neuron : layer->Neurons) {
      delete neuron;
    }
    delete layer;
  }
}

void MFNetwork::DoLayerAdd(int neurons) {
  Layer_t *layer = new Layer_t();
  
  // Populate layer with neurons.
  for (int i = 0; i < neurons; ++i) {
    layer->Neurons.push_back(new Neuron());
  }

  // Routing map setup.
  if (layers_.size() == 1) {
    // The output layer needs a special routing map, because its output is going
    // back to the user.
    for (uint32_t i = 0; i < num_outputs_; ++i) {
      layer->RoutingMap[i] = std::vector<int> (1, i);
    }
  } else {
    // Any other layer defaults to having each neuron send its outputs to every
    // neuron in the next layer.
    uint32_t size;
    if (layers_.empty()) {
      // Input layer.
      size = num_inputs_;
    } else {
      // Hidden layer.
      size = layer_size_;
    }

    for (uint32_t i = 0; i < size; ++i) {
      std::vector<int> destinations;
      for (uint32_t i1 = 0; i1 < layer_size_; ++i1) {
        destinations.push_back(i1);
      }
      layer->RoutingMap[i] = destinations;
    }
  }

  // Add it to the list of layers.
  if (layers_.size() <= 1) {
    // For the input and output, we can push back normally.
    layers_.push_back(layer);
  } else {
    // Insert right before our output layer.
    auto it = layers_.end();
    --it;
    layers_.insert(it, layer);
  }
}

void MFNetwork::SetInputs(double *values) {
  // Write to our buffer. (It will get sent to the input layer later.)
  layer_input_buffer_.clear();
  for (uint32_t i = 0; i < num_inputs_; ++i) {
    layer_input_buffer_[i].push_back(values[i]);
  }
}
  
bool MFNetwork::DoGetOutputs(double *values, std::vector<double> *osubj) {
  bool save_outputs = true;
  if (osubj == nullptr) {
    // We're not doing back propagation.
    save_outputs = false;
  }

  // Maps the output of each neuron in a layer to the index of its neuron.
  std::map<int, double> layer_output_buffer;

  // Calculate each layer in sequence.
  for (Layer_t *layer : layers_) {
    for (uint32_t neuron_i = 0; neuron_i < layer->Neurons.size(); ++neuron_i) {
      Neuron *neuron = layer->Neurons[neuron_i];
      // Set the inputs that we're using for this neuron.
      neuron->SetInputs(layer_input_buffer_[neuron_i]);
      
      if (use_random_) {
        // We might need new random weights for our neuron if the number of
        // inputs has changed.
        std::vector<double> weights;
        neuron->GetWeights(&weights);
        // We're going to try to change as few of the current weights as
        // possible.
        while (weights.size() > layer_input_buffer_[neuron_i].size()) {
          weights.pop_back();
        }
        while (weights.size() < layer_input_buffer_[neuron_i].size()) {
          int range = upper_ - lower_;
          int num = rand() % range + lower_;
          weights.push_back(num);
        }
        neuron->SetWeights(weights);
      }
      
      double out;
      if (!neuron->GetOutput(&out)) {
        return false;
      }
      layer_output_buffer[neuron_i] = out;
      if (save_outputs) {
        osubj->push_back(out);
      }
    }  

    // First clear all the output vectors.
    layer_input_buffer_.clear();
    // Send our outputs to our inputs for the next layer, using the layer
    // routing map to keep track of which output goes where.
    for (auto& kv : layer_output_buffer) {
      std::vector<int> destinations = layer->RoutingMap[kv.first];
      for (int dest : destinations) {
        layer_input_buffer_[dest].push_back(kv.second);
      }
    }
    layer_output_buffer.clear();
  }

  // Check that we have the expected number of outputs.
  // We use layer_input_buffer_ here since layer_output_buffer has been cleared.
  ASSERT(layer_input_buffer_.size() == num_outputs_,
      "Got the wrong number of outputs.");
  for (uint32_t i = 0; i < num_outputs_; ++i) {
    // Should be insured by special routing map for output layer.
    ASSERT(layer_input_buffer_[i].size() == 1,
        "Invalid routing for output layer.");
    values[i] = layer_input_buffer_[i][0];
    if (save_internal) {
      // We don't want the last outputs in our vector of internal outputs.
      osubj->pop_back();
    }
  }

  return true;
}

Neuron *MFNetwork::GetNeuron(uint32_t layer_i, uint32_t neuron_i) {
  // Do a little range checking to guard for user stupidity.
  if (layer_i >= layers_.size()) {
    return nullptr;
  }
  Layer_t *layer = layers_[layer_i];
  if (neuron_i >= layer->Neurons.size()) {
    return nullptr;
  }

  return layer->Neurons[neuron_i];
}

void MFNetwork::SetWeights(const std::vector<double>& values) {
  for (uint32_t i = 0; i < layers_.size(); ++i) {
    ASSERT(SetLayerWeights(i, values), 
        "SetLayerWeights failing for a weird reason.");
  }
}

bool MFNetwork::SetLayerWeights(uint32_t layer_i, const std::vector<double>& values) {
  if (layer_i >= layers_.size()) {
    return false;
  }
  Layer_t *layer = layers_[layer_i];
  for (Neuron *neuron : layer->Neurons) {
    neuron->SetWeights(values);
  }
  return true;
}

void MFNetwork::SetOutputFunctions(ImpulseFunction *impulse) {
  for (Layer_t *layer : layers_) {
    for (Neuron *neuron : layer->Neurons) {
      neuron->SetOutputFunction(impulse);
    }
  }
}

bool MFNetwork::SetLayerOutputFunctions(uint32_t layer_i,
    ImpulseFunction *impulse) {
  if (layer_i >= layers_.size()) {
    return false;
  }
  Layer_t *layer = layers_[layer_i];
  for (Neuron *neuron : layer->Neurons) {
    neuron->SetOutputFunction(impulse);
  }
  return true;
}

bool MFNetwork::SetOutputRoute(uint32_t layer_i, uint32_t neuron_i,
    const std::vector<int>& output_nodes) {
  if (layer_i >= layers_.size()) {
    return false;
  }
  Layer_t *layer = layers_[layer_i];
  if (neuron_i >= layer->Neurons.size()) {
    return false;
  }
  
  // Write to the proper layer's routing map.
  layer->RoutingMap[neuron_i] = output_nodes;
  return true;
}

bool MFNetwork::CopyLayout(const MFNetwork& source) {
  // First, some sanity checks.
  if (source.num_inputs_ != num_inputs_ ||
      source.num_outputs_ != num_outputs_ ||
      source.layer_size_ != layer_size_) {
    return false;
  }

  // Basically, we need to copy the routing maps from each layer.
  for (uint32_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->RoutingMap = source.layers_[i]->RoutingMap;
  }

  return true;
}

bool MFNetwork::PropagateError(double *targets) {
  double outputs [num_outputs_];
  std::vector<double> internal;
  DoGetOutputs(outputs, &internal);
  double error [num_outputs_];
  for (uint32_t i = 0; i < num_outputs_; ++i) {
    error[i] = targets[i] - outputs[i];
  }

  // Iterate across our network backwards.
  // Map our signals to neuron indices.
  std::map<uint32_t, double> signals;
  for (uint32_t layer_i = layers_.size() - 1; layer_i >= 0; --layer_i) {
    Layer_t *layer = layers_[layer_i];
    for (uint32_t neuron_i = layer->Neurons.size() - 1; 
        neuron_i >= 0; --neuron_i) {
      Neuron *neuron = layer->Neurons[neuron_i];
      ImpulseFunction *impulse = neuron->GetOutputFunction();
      
      if (impulse != nullptr) {
        if (layer_i == layers_.size() - 1) {
          // Output layer.
          double signal = impulse->Derivative(error[neuron_i]) * error[neuron_i];
          signals[neuron_i] = signal;
          neuron->AdjustWeights(learning_rate_, signal);
        } else {
          // Hidden layer.
          // Based on how we assign outputs to weights, we can work backwards to
          // find which weight one the downstream neuron governs the output of
          // this one.
          double error = 0;
          for (int )
        }
      } else {
        return false;
      }
    }
  }
}

bool MFNetwork::Clone(MFNetwork *dest) {
  // Make one with the same specifications.
  dest = new MFNetwork(num_inputs_, num_outputs_, layer_size_);
  // Copy the routing map.
  dest->CopyLayout(*this);
  return true;
}

size_t MFNetwork::GetChromosomeSize() {
  int num_weights = 0;
  for (Layer_t *layer : layers_) {
    for (Neuron *neuron : layer->Neurons) {
      num_weights += neuron->GetNumWeights(); 
    }
  }

  return num_weights;
}

bool MFNetwork::GetChromosome(uint64_t *chromosome) {
  std::vector<double> neuron_weights;
  int weights_i = 0;
  for (Layer_t *layer : layers_) {
    for (Neuron *neuron : layer->Neurons) {
      neuron->GetWeights(&neuron_weights);
      for (double weight : neuron_weights) {
        memcpy(&chromosome[weights_i++], &weight, sizeof(double));
      }
    }
  }
  return true;
}

bool MFNetwork::SetChromosome(uint64_t *chromosome) {
  int weights_i = 0;
  for (Layer_t *layer : layers_) {
    for (Neuron *neuron : layer->Neurons) {
      int weights_num = neuron->GetNumWeights();
      int max_index = weights_i + weights_num - 1;
      std::vector<double> weights_v;
      for (; weights_i <= max_index; ++weights_i) {
        double weight;
        memcpy(&weight, &chromosome[weights_i], sizeof(uint64_t));
        weights_v.push_back(weight);
      }
      neuron->SetWeights(weights_v);
    }
  }
  return true;
}

bool MFNetwork::SaveToFile(const char *path, MFNetwork *network) {
  FILE *out_file = fopen(path, "w");
  if (!out_file) {
    return false;
  }

  fwrite(network, sizeof(*network), 1, out_file);
  // Save the size of our chromosome and our actual chromosome so we can recover
  // our weights later.
  uint32_t size = network->GetChromosomeSize();
  fwrite(&size, sizeof(size), 1, out_file);
  uint64_t chromosome [size];
  network->GetChromosome(chromosome);
  fwrite(chromosome, sizeof(chromosome[0]), size, out_file);
  fclose(out_file);

  return true;
}

bool MFNetwork::ReadFromFile(const char *path, MFNetwork *network) {
  FILE *in_file = fopen(path, "r");
  if (!in_file) {
    return false;
  }

  fread(network, sizeof(*network), 1, in_file);
  uint32_t size;
  fread(&size, sizeof(size), 1, in_file);
  uint64_t chromosome [size];
  fread(chromosome, sizeof(chromosome[0]), size, in_file);
  fclose(in_file);

  // Create new neurons and layers for our network, as the old pointers are
  // invalid.
  for (uint32_t i = 0; i < network->layers_.size(); ++i) {
    size_t layer_size;
    if (i == 0) {
      // Input layer.
      layer_size = network->num_inputs_;
    } else if (i == network->layers_.size() - 1) {
      // Output layer.
      layer_size = network->num_outputs_;
    } else {
      // Hidden layer.
      layer_size = network->layer_size_;
    }
    network->layers_[i] = new Layer_t;
    for (uint32_t i1 = 0; i1 < layer_size; ++i1) {
      network->layers_[i]->Neurons.push_back(new Neuron);
    }
  }
  // Set our neuron weights.
  network->SetChromosome(chromosome);

  return true;
}

} //network
