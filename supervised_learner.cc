#include <math.h>
#include <stdio.h> // temp
#include <unistd.h> // temp

#include <algorithm>
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
  int cycle = 0;

  // Separate out training and testing data.
  uint32_t split = training_data_.size() * 0.8;
  std::vector<TrainingItem> training_data;
  std::vector<TrainingItem> testing_data;
  std::random_shuffle(training_data_.begin(), training_data_.end());
  if (training_data_.size() == 1) {
    training_data = testing_data = training_data_;
  } else {
    for (uint32_t i = 0; i < training_data_.size(); ++i) {
      if (i < split) {
        training_data.push_back(training_data_[i]);
      } else {
        testing_data.push_back(training_data_[i]);
      }
    }
  }
  while (max_iterations == -1 || cycle < max_iterations) {
    current_error = 0;
    // Set input based on training data.
    std::random_shuffle(training_data.begin(), training_data.end());
    for (TrainingItem item : training_data) {
      trainee_->SetInputs(item.InputData);
      // Get network output.
      double outputs [num_outputs_];
      std::vector<double> internal;
      if (!trainee_->DoGetOutputs(outputs, &internal)) {
        return false;
      }
      //printf("Training Output: %f\n", outputs[0]);
      
      // Do BackPropagation
      if (!trainee_->PropagateError(item.ExpectedOutput, outputs, &internal)) {
        return false;
      }
    }

    for (TrainingItem item : testing_data) {
      trainee_->SetInputs(item.InputData);
      double outputs [num_outputs_];
      if (!trainee_->GetOutputs(outputs)) {
        return false;
      }
      //printf("Output: %f\n", outputs[0]);
      // Calculate cumulative error.
      for (uint32_t i = 0; i < num_outputs_; ++i) {
        current_error += pow(item.ExpectedOutput[i] - outputs[i], 2);
      }
    }
    current_error /= 2;
    printf("Total Error: %f\n", current_error);
    if (current_error < error) {
      break;
    }
    
    ++cycle;
    printf("Cycle: %d\n", cycle);

    //sleep(1);
  }
  return true;
}

} // algorithm
