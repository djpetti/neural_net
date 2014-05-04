// Tests for SupervisedLearner.

#include "gtest/gtest.h"
#include "../multilayered_feedforward.h"
#include "../output_functions.h"
#include "../supervised_learner.h"

namespace algorithm {
namespace test {

TEST(BasicTests, SinglePoint) {
  // Can we train for single criterion?
  // This test doesn't really fail, it either gets stuck in a loop or segfaults.
  network::MFNetwork network(1, 1, 5);
  network.AddHiddenLayer();
  network.AddHiddenLayer();
  network::Sigmoid sigmoid;
  network.RandomWeights(-50, 50);
  network.SetOutputFunctions(&sigmoid);
  SupervisedLearner learner(&network);

  double input [] = {0.01};
  double target [] = {0.5};
  learner.AddTrainingData(input, target);

  EXPECT_TRUE(learner.Learn(0.00001));
}

} // test
} // algorithm
