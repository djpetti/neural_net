#ifndef SUPERVISED_LEARNER_H_
#define SUPERVISED_LEARNER_H_

#include <stdint.h>

#include <vector>

#include "macros.h"
#include "multilayered_feedforward.h"

namespace algorithm {

// A class that provides a nice implementation for supervised learning via
// backpropagation.
class SupervisedLearner {
 public:
  // Ctor argument specifies a network to train.
  explicit SupervisedLearner(network::MFNetwork *trainee);
  ~SupervisedLearner();
  // Add to the networks training set. <input> is an array containing what will
  // be fed to the inputs of the network, and <output> is the expected output.
  void AddTrainingData(double *input, double *output);
  // Runs backpropagation iterations until the error is less than <error>, or
  // <max_iterations> iterations have been performed.
  bool Learn(double error, int max_iterations = -1);

  DISSALOW_COPY_AND_ASSIGN(SupervisedLearner);

 private:
  struct TrainingItem {
    // The actual data getting fed into the network.
    double *InputData;
    // The expected output for this input data.
    double *ExpectedOutput;
  };

  std::vector<TrainingItem> training_data_;
  network::MFNetwork *trainee_;
  uint32_t num_inputs_, num_outputs_;
};

} // algorithm

#endif
