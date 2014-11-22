#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "logger.h"
#include "multilayered_feedforward.h"

namespace network {

static_assert(sizeof(double) == sizeof(uint64_t),
            "Otherwise weird things happen.");

MFNetwork::MFNetwork(uint32_t inputs, uint32_t outputs, uint32_t layer_size) :
    num_inputs_(inputs),
    num_outputs_(outputs),
    layer_size_(layer_size),
    use_special_weights_(0),
    learning_rate_(0.01),
    momentum_(0.5) {
  // Seed random number generator.
  srand(time(NULL));
  // Create the input and output layers.
  AddHiddenLayer(num_inputs_);
  AddHiddenLayer(num_outputs_);
}

MFNetwork::~MFNetwork() {
  for (Layer_t *layer : layers_) {
    for (Neuron *neuron : layer->Neurons) {
      delete neuron;
    }
    delete layer;
  }
}

void MFNetwork::AddHiddenLayer(int size/* = -1*/) {
  if (size < 0) {
    size = layer_size_;
  }

  Layer_t *layer = new Layer_t();

  // Populate layer with neurons.
  std::vector<double> ones(1, 1);
  for (int i = 0; i < size; ++i) {
    Neuron *neuron = new Neuron();
    if (!layers_.size()) {
      // All the weights in our input layer should always be one.
      neuron->SetWeights(ones);
    }
    layer->Neurons.push_back(neuron);
  }

  // Routing map setup.
  if (layers_.size() == 1) {
    // The output layer needs a special routing map, because its output is going
    // back to the user.
    for (uint32_t i = 0; i < num_outputs_; ++i) {
      layer->RoutingMap[i] = std::vector<int> (1, i);
    }
    // Also, the input should be connected directly to the outputs.
    UpdateRouting(layers_[0], layer);
  } else if (!layers_.empty()) {
    // Hidden layers default to having each neuron send its outputs to every
    // neuron in the next layer.
    UpdateRouting(layer, layers_[layers_.size() - 1]);
  }

  if (layers_.size() >= 2) {
    // If we're using default routing, we need to modify the penultimate
    // layer so that it broadcasts to the right number of neurons for the next
    // hidden layer instead of the output layer.
    Layer_t *last_added = layers_[layers_.size() - 2];
    if (last_added->DefaultRouting) {
      UpdateRouting(last_added, layer);
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

bool MFNetwork::RemoveLayer(uint32_t index) {
  if (!index || index >= layers_.size() - 1) {
    LOG(Level::WARNING, "Cannot remove layer at index %d.");
    return false;
  }

  Layer_t *to_delete = layers_[index];
  layers_.erase(layers_.begin() + index);
  delete to_delete;

  return true;
}

void MFNetwork::SetInputs(const double *values) {
  // Write to our buffer. (It will get sent to the input layer later.)
  layer_input_buffer_.clear();
  for (uint32_t i = 0; i < num_inputs_; ++i) {
    layer_input_buffer_[i].push_back(values[i]);
  }
}

bool MFNetwork::DoUpdate(double *values) {
  if (!HiddenLayerQuantity()) {
    // Our network won't work without at least one hidden layer, and it will be
    // pretty useless.
    return false;
  }

  // Maps the output of each neuron in a layer to the index of its neuron.
  std::map<int, double> layer_output_buffer;

  // Calculate each layer in sequence.
  for (uint32_t layer_i = 0; layer_i < layers_.size(); ++layer_i) {
    Layer_t *layer = layers_[layer_i];
    for (uint32_t neuron_i = 0; neuron_i < layer->Neurons.size(); ++neuron_i) {
      Neuron *neuron = layer->Neurons[neuron_i];
      // Set the inputs that we're using for this neuron.
      if (values) {
        neuron->SetInputs(layer_input_buffer_[neuron_i]);
      }

      if (use_special_weights_) {
        // We might need new weights for our neuron if the number of
        // inputs has changed.
        std::vector<double> weights;
        neuron->GetWeights(&weights);
        // We're going to try to change as few of the current weights as
        // possible.
        while (weights.size() > layer_input_buffer_[neuron_i].size()) {
          weights.pop_back();
        }
        while (weights.size() < layer_input_buffer_[neuron_i].size()) {
          double num;
          if (use_special_weights_ == 1) {
            // Random weights.
            int range = upper_ * 1000 - lower_ * 1000;
            num = rand() % range + lower_ * 1000;
            num /= 1000;
          } else {
            // User-specified weights.
            num = user_weight_;
          }
          if (!layer_i) {
            // All the weights of the input layer should be 1.
            num = 1;
          }
          weights.push_back(num);
        }
        neuron->SetWeights(weights);
      }

      if (values) {
        // If we're just setting weights, there's no point in wasting time doing
        // this.
        double out;
        if (!neuron->GetOutput(&out)) {
          return false;
        }
        layer_output_buffer[neuron_i] = out;
      } else {
        // Just put zeroes into out output buffer.
        layer_output_buffer[neuron_i] = 0;
      }
    }

    // First clear all the output vectors.
    layer_input_buffer_.clear();
    // Send our outputs to our inputs for the next layer, using the layer
    // routing map to keep track of which output goes where.
    for (auto& kv : layer_output_buffer) {
      for (int dest : layer->RoutingMap[kv.first]) {
        layer_input_buffer_[dest].push_back(kv.second);
      }
    }
    layer_output_buffer.clear();
  }

  // Check that we have the expected number of outputs.
  // We use layer_input_buffer_ here since layer_output_buffer has been cleared.
  CHECK(layer_input_buffer_.size() == num_outputs_,
      "Got the wrong number of outputs.");
  for (uint32_t i = 0; i < num_outputs_; ++i) {
    // Should be insured by special routing map for output layer.
    CHECK(layer_input_buffer_[i].size() == 1,
        "Invalid routing for output layer.");
    if (values) {
      values[i] = layer_input_buffer_[i][0];
    }
  }

  initialized_ = true;
  return true;
}

bool MFNetwork::CheckInitialized() {
  if (initialized_) {
    return true;
  }
  if (use_special_weights_) {
    // In this case, simply forcing weight updating will make us initialized.
    LOG(Level::DEBUG, "CheckInitialized(): forcing weight update.");
    return ForceWeightUpdate();
  } else {
    // Check that all our neurons have weights set. (Even if it's not the
    // correct number of weights, it will still be able to write the network to
    // and from a file and set/get it's chromosome correctly, which is all that
    // we care about here.)
    for (auto layer : layers_) {
      for (auto neuron : layer->Neurons) {
        if (!neuron->GetNumWeights()) {
          return false;
        }
      }
    }

    initialized_ = true;
    return true;
  }
}

Neuron *MFNetwork::GetNeuron(uint32_t layer_i, uint32_t neuron_i) {
  // Do a little range checking to guard for user stupidity.
  if (!layer_i || layer_i >= layers_.size()) {
    return nullptr;
  }
  Layer_t *layer = layers_[layer_i];
  if (neuron_i >= layer->Neurons.size()) {
    return nullptr;
  }

  return layer->Neurons[neuron_i];
}

bool MFNetwork::SetLayerWeights(uint32_t layer_i,
    const std::vector<double>& values) {
  if (!layer_i || layer_i >= layers_.size()) {
    return false;
  }
  Layer_t *layer = layers_[layer_i];
  for (Neuron *neuron : layer->Neurons) {
    neuron->SetWeights(values);
  }
  return true;
}

void MFNetwork::SetOutputFunctions(ImpulseFunction *impulse) {
  for (uint32_t layer_i = 1; layer_i < layers_.size(); ++layer_i) {
    Layer_t *layer = layers_[layer_i];
    for (Neuron *neuron : layer->Neurons) {
      neuron->SetOutputFunction(impulse);
    }
  }
}

bool MFNetwork::SetLayerOutputFunctions(uint32_t layer_i,
    ImpulseFunction *impulse) {
  if (!layer_i || layer_i >= layers_.size()) {
    return false;
  }
  Layer_t *layer = layers_[layer_i];
  for (Neuron *neuron : layer->Neurons) {
    neuron->SetOutputFunction(impulse);
  }
  return true;
}

void MFNetwork::SetBiases(double bias) {
  for (uint32_t i = 1; i < layers_.size(); ++i) {
    CHECK(SetLayerBiases(i, bias),
        "SetLayerBiases failing for some weird reason.");
  }
}

bool MFNetwork::SetLayerBiases(uint32_t layer_i, double bias) {
  if (!layer_i || layer_i >= layers_.size()) {
    return false;
  }

  Layer_t *layer = layers_[layer_i];
  for (Neuron *neuron : layer->Neurons) {
    neuron->SetBias(bias);
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
  layer->DefaultRouting = false;
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

bool MFNetwork::PropagateError(const double *targets,
    double *final_outputs/* = nullptr*/) {
  double outputs [num_outputs_];
  std::vector<double> internal;
  if (!final_outputs) {
    GetOutputs(outputs);
  } else {
    // We can skip calculating outputs.
    memcpy(outputs, final_outputs, sizeof(outputs[0]) * num_outputs_);
  }

  // Stores the errors from the last layer. We also use it to store the network
  // errors initially.
  std::map<uint32_t, double> last_errors_input;
  // Buffer for last_errors_input.
  std::map<uint32_t, double> last_errors_output;
  for (uint32_t i = 0; i < num_outputs_; ++i) {
    last_errors_input[i] = targets[i] - outputs[i];
  }

  // Iterate across our network backwards.
  for (int layer_i = layers_.size() - 1; layer_i >= 0; --layer_i) {
    Layer_t *layer = layers_[layer_i];
    for (int neuron_i = layer->Neurons.size() - 1;
        neuron_i >= 0; --neuron_i) {
      Neuron *neuron = layer->Neurons[neuron_i];
      ImpulseFunction *impulse = neuron->GetOutputFunction();

      if (impulse != nullptr) {
        double error = 0;
        if (static_cast<uint32_t>(layer_i) == layers_.size() - 1) {
          // Output layer.
          error = last_errors_input[neuron_i];
        } else if (layer_i != 0) {
          // Hidden layer.
          // Based on how we assign outputs to weights, we can work backwards to
          // find which weight on the downstream neuron governs the output of
          // this one.
          Layer_t *lower_layer = layers_[layer_i + 1];
          for (int dest : layer->RoutingMap[neuron_i]) {
            Neuron *lower_neuron = lower_layer->Neurons[dest];
            double weight;
            CHECK(lower_neuron->GetLastWeight(&weight),
                "Neuron has the wrong number of weights.");
            error += weight * last_errors_input[dest];
          }
       }

       // Do this as long as it's not the input layer.
       if (layer_i != 0) {
         last_errors_output[neuron_i] = error;
         CHECK(neuron->AdjustWeights(learning_rate_, momentum_, error),
             "Failed to update neuron weights.");
        }
      } else {
        return false;
      }
    }

    // Swap the last errors buffers for a new cycle.
    last_errors_input.swap(last_errors_output);
    last_errors_output.clear();
  }

  return true;
}

bool MFNetwork::Clone(MFNetwork *dest) {
  // Make one with the same specifications.
  dest = new MFNetwork(num_inputs_, num_outputs_, layer_size_);
  // Copy the routing map.
  dest->CopyLayout(*this);
  return true;
}

uint32_t MFNetwork::GetNeuronQuantity() {
  uint32_t total = 0;
  for (Layer_t *layer : layers_) {
    total += layer->Neurons.size();
  }
  return total;
}

size_t MFNetwork::GetChromosomeSize() {
  if (!CheckInitialized()) {
    return 0;
  }

  int num_weights = 0;
  for (uint32_t layer_i = 1; layer_i < layers_.size(); ++layer_i) {
    Layer_t *layer = layers_[layer_i];
    for (Neuron *neuron : layer->Neurons) {
      num_weights += neuron->GetNumWeights();
      ++num_weights;
    }
  }

  return num_weights;
}

bool MFNetwork::GetChromosome(uint64_t *chromosome) {
  if (!CheckInitialized()) {
    return false;
  }

  std::vector<double> neuron_weights;
  int weights_i = 0;
  for (uint32_t layer_i = 1; layer_i < layers_.size(); ++layer_i) {
    Layer_t *layer = layers_[layer_i];
    for (Neuron *neuron : layer->Neurons) {
      neuron->GetWeights(&neuron_weights);
      for (double weight : neuron_weights) {
        // The fancy memcpy-ing is due to the type mismatch.
        memcpy(&chromosome[weights_i++], &weight, sizeof(weight));
      }
      // Bias weight.
      double bias = neuron->GetBias();
      memcpy(&chromosome[weights_i++], &bias, sizeof(bias));
    }
  }
  return true;
}

bool MFNetwork::SetChromosome(uint64_t *chromosome) {
  // Zero all our weights and force a network update so that we can make sure
  // that GetNumWeights() will return the right number.
  SetWeights(0);
  if (!ForceWeightUpdate()) {
    return false;
  }

  int weights_i = 0;
  for (uint32_t layer_i = 1; layer_i < layers_.size(); ++layer_i) {
    Layer_t *layer = layers_[layer_i];
    for (Neuron *neuron : layer->Neurons) {
      const int weights_num = neuron->GetNumWeights();
      const int max_index = weights_i + weights_num - 1;
      std::vector<double> weights_v;
      for (; weights_i <= max_index; ++weights_i) {
        double weight;
        memcpy(&weight, &chromosome[weights_i], sizeof(weight));
        weights_v.push_back(weight);
      }
      neuron->SetWeights(weights_v);
      // Bias weight.
      double bias;
      memcpy(&bias, &chromosome[weights_i++], sizeof(bias));
      neuron->SetBias(bias);
    }
  }
  return true;
}

void MFNetwork::SerializeRoutes(uint32_t *routes) {
  int index = 0;
  for (Layer_t *layer : layers_) {
    routes[index++] = layer->RoutingMap.size();
    routes[index++] = layer->DefaultRouting;
    for (auto& kv : layer->RoutingMap) {
      routes[index++] = kv.first;
      routes[index++] = kv.second.size();
      for (int neuron_i : kv.second) {
        routes[index++] = neuron_i;
      }
    }
  }
}

size_t MFNetwork::GetNumRoutes() {
  size_t total = 0;
  for (Layer_t *layer : layers_) {
    // For the size of the routing map and the value of DefaultRouting.
    total += 2;
    for (auto& kv : layer->RoutingMap) {
      // For first value and size of vector.
      total += 2;
      // For vector contents.
      total += kv.second.size();
    }
  }
  return total;
}

void MFNetwork::DeserializeRoutes(uint32_t *routes) {
  int index = 0;
  for (Layer_t *layer : layers_) {
    int map_size = routes[index++];
    layer->DefaultRouting = routes[index++];
    for (int i = 0; i < map_size; ++i) {
      int source_neuron_i = routes[index++];
      int dest_size = routes[index++];
      std::vector<int> destinations;
      for (int i1 = 0; i1 < dest_size; ++i1) {
        destinations.push_back(routes[index++]);
      }
      layer->RoutingMap[source_neuron_i] = destinations;
    }
  }
}

void MFNetwork::UpdateRouting(Layer_t *source, Layer_t *dest) {
  source->RoutingMap.clear();
  for (uint32_t i = 0; i < source->Neurons.size(); ++i) {
    for (uint32_t i1 = 0; i1 < dest->Neurons.size(); ++i1) {
      source->RoutingMap[i].push_back(i1);
    }
  }
}

bool MFNetwork::SaveToFile(const char *path) {
  if (!CheckInitialized()) {
    return false;
  }

  FILE *out_file = fopen(path, "w");
  if (!out_file) {
    return false;
  }

  // Save basic network information to the file.
  int32_t basic_info [] = {
      static_cast<int32_t>(num_inputs_),
      static_cast<int32_t>(num_outputs_),
      static_cast<int32_t>(layer_size_),
      static_cast<int32_t>(use_special_weights_),
      static_cast<int32_t>(initialized_),
      upper_,
      lower_ };
  const size_t basic_info_size = sizeof(basic_info) / sizeof(int32_t);
  double weight_info[] = {
      user_weight_};
  const size_t weight_info_size = sizeof(weight_info) / sizeof(double);
  fwrite(basic_info, sizeof(basic_info[0]), basic_info_size, out_file);
  fwrite(weight_info, sizeof(weight_info[0]), weight_info_size, out_file);

  // Save hidden layer size information.
  uint32_t num_hidden = HiddenLayerQuantity();
  uint32_t layer_sizes[num_hidden];
  for (uint32_t i = 0; i < num_hidden; ++i) {
    layer_sizes[i] = layers_[i + 1]->Neurons.size();
  }
  fwrite(&num_hidden, sizeof(num_hidden), 1, out_file);
  if (num_hidden > 0) {
    fwrite(layer_sizes, sizeof(layer_sizes[0]), num_hidden, out_file);
  }

  // Save our routing information from the layers structures.
  uint32_t num_routes = GetNumRoutes();
  fwrite(&num_routes, sizeof(num_routes), 1, out_file);
  uint32_t routes [num_routes];
  SerializeRoutes(routes);
  fwrite(routes, sizeof(int), num_routes, out_file);

  // Save the size of our chromosome and our actual chromosome so we can recover
  // our weights later.
  uint32_t size = GetChromosomeSize();
  fwrite(&size, sizeof(size), 1, out_file);
  uint64_t chromosome [size];
  GetChromosome(chromosome);
  fwrite(chromosome, sizeof(chromosome[0]), size, out_file);

  fclose(out_file);

  return true;
}

bool MFNetwork::ReadFromFile(const char *path) {
  FILE *in_file = fopen(path, "r");
  if (!in_file) {
    return false;
  }

  // Retrieve basic info.
  constexpr size_t basic_info_size = 7;
  constexpr size_t weight_info_size = 1;
  int32_t basic_info [basic_info_size];
  double weight_info [weight_info_size];
  fread(basic_info, sizeof(basic_info[0]), basic_info_size, in_file);
  fread(weight_info, sizeof(weight_info[0]), weight_info_size, in_file);
  int basic_info_index = 0;
  num_inputs_ = basic_info[basic_info_index++];
  num_outputs_ = basic_info[basic_info_index++];
  layer_size_ = basic_info[basic_info_index++];
  use_special_weights_ = basic_info[basic_info_index++];
  initialized_ = basic_info[basic_info_index++];
  upper_ = basic_info[basic_info_index++];
  lower_ = basic_info[basic_info_index++];
  int weight_info_index = 0;
  user_weight_ = weight_info[weight_info_index++];

  // Retrieve hidden layer sizes.
  uint32_t num_hidden;
  fread(&num_hidden, sizeof(num_hidden), 1, in_file);
  uint32_t layer_sizes [num_hidden];
  if (num_hidden > 0) {
    fread(layer_sizes, sizeof(layer_sizes[0]), num_hidden, in_file);
  }

  // Our ctor already added two layers, but possibly without knowing the correct
  // size parameters. Therefore, we'll redo them.
  layers_.clear();
  AddHiddenLayer(num_inputs_);
  AddHiddenLayer(num_outputs_);
  for (uint32_t i = 0; i < num_hidden; ++i) {
    AddHiddenLayer(layer_sizes[i]);
  }

  // Retrieve routing info.
  uint32_t num_routes;
  fread(&num_routes, sizeof(num_routes), 1, in_file);
  uint32_t routes [num_routes];
  fread(routes, sizeof(routes[0]), num_routes, in_file);

  // Retrieve chromosome.
  uint32_t size;
  fread(&size, sizeof(size), 1, in_file);
  uint64_t chromosome [size];
  fread(chromosome, sizeof(chromosome[0]), size, in_file);

  fclose(in_file);

  // Set our neuron weights and layer routings.
  DeserializeRoutes(routes);
  SetChromosome(chromosome);

  return true;
}

} //network
