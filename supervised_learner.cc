#include <math.h>
#include <stdio.h> // TEMP

#include <limits>

#include "supervised_learner.h"

namespace algorithm {

SupervisedLearner::SupervisedLearner(network::MFNetwork *trainee) :
    trainee_(trainee),
    num_inputs_(trainee->num_inputs_),
    num_outputs_(trainee->num_outputs_) {}

SupervisedLearner::~SupervisedLearner() {
  for (TrainingItem item : training_data_) {
    delete[] item.InputData;
    delete[] item.ExpectedOutput;
  }
}

void SupervisedLearner::AddTrainingData(double *input, double *output) {
  TrainingItem item;
  // Allocate arrays.
  item.InputData = new double[num_inputs_];
  item.ExpectedOutput = new double[num_outputs_];
  memcpy(item.InputData, input, sizeof(input[0]) * num_inputs_);
  memcpy(item.ExpectedOutput, output, sizeof(output[0]) * num_outputs_);
  
  training_data_.push_back(item);
}

bool SupervisedLearner::Learn(double error, int max_iterations/* = -1*/) {
  double current_error = std::numeric_limits<double>::max();
  int index = 0;
  while (max_iterations == -1 || index < max_iterations) {
    current_error = 0;
    // Set input based on training data.
    for (TrainingItem item : training_data_) {
      trainee_->SetInputs(item.InputData);
      // Get network output.
      double outputs [num_outputs_];
      std::vector<double> internal;
      if (!trainee_->DoGetOutputs(outputs, &internal)) {
        return false;
      }
      
      // Calculate cumulative error.
      for (uint32_t i = 0; i < num_outputs_; ++i) {
        current_error += pow(item.ExpectedOutput[i] - outputs[i], 2);
      }
      // Do BackPropagation
      if (!trainee_->PropagateError(item.ExpectedOutput, outputs, &internal)) {
        return false;
      }
     
      ++index;
    }
    current_error /= 2;
    if (current_error < error) {
      break;
    }
  }
  return true;
}

} // algorithm
