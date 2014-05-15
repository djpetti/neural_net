// Tests for SupervisedLearner.
#include <math.h>

#include "gtest/gtest.h"
#include "../multilayered_feedforward.h"
#include "../output_functions.h"
#include "../supervised_learner.h"

namespace algorithm {
namespace test {

TEST(BasicTests, SinglePoint) {
  // Can we train for single criterion?
  network::MFNetwork network(1, 1, 5);
  network.AddHiddenLayer();
  network::Sigmoid sigmoid;
  network.RandomWeights(-1, 1);
  network.SetOutputFunctions(&sigmoid);
  network.SetMomentum(0.01);
  SupervisedLearner learner(&network);

  double input [] = {0.01};
  double target [] = {0.5};
  learner.AddTrainingData(input, target);

  EXPECT_TRUE(learner.Learn(0.0001));
}

TEST(BasicTests, SineWaveTest) {
  // A much more complicated test that attempts to approximate a sine wave.
  network::MFNetwork network(1, 1, 14);
  network.AddHiddenLayers(1);
  network::Sigmoid sigmoid;
  network::Linear linear(1);
  network.RandomWeights(-2, 2);
  network.SetOutputFunctions(&sigmoid);
  network.SetLayerOutputFunctions(2, &linear);
  network.SetLearningRate(0.2);
  network.SetMomentum(0);
  SupervisedLearner learner(&network);

  // Add training data based on our sine wave. We'll break it into 20
  // subintervals.
  double input [1];
  double output [1];
  for (int i = 1; i <= 20; ++i) {
    input[0] = (4 * M_PI / 20) * i;
    output[0] = sin(input[0]);
    learner.AddTrainingData(input, output);
  }

  EXPECT_TRUE(learner.Learn(0.03, 15000));
  
  double actual [1];
  network.SetInputs(input);
  network.GetOutputs(actual);
}

} // test
} // algorithm
