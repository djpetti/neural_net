// Tests for SupervisedLearner.
#include <math.h>
#include <stdio.h> // temp

#include "gtest/gtest.h"
#include "../multilayered_feedforward.h"
#include "../output_functions.h"
#include "../supervised_learner.h"

namespace algorithm {
namespace test {

/*TEST(BasicTests, SinglePoint) {
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
}*/

TEST(BasicTests, SineWaveTest) {
  // A much more complicated test that attempts to approximate a sine wave.
  // Note that sigmoid-activated networks are not very good at this, that's why
  // ours is so huge.
  network::MFNetwork network(1, 1, 14);
  network.AddHiddenLayers(1);
  network::Sigmoid sigmoid;
  network::Linear linear(1);
  network::DumbOutputer dumb;
  network.RandomWeights(-2, 2);
  network.SetOutputFunctions(&sigmoid);
  network.SetLayerOutputFunctions(2, &linear);
  network.SetLayerOutputFunctions(0, &dumb);
  network.SetLearningRate(0.2);
  network.SetMomentum(0);
  SupervisedLearner learner(&network);

  // Add training data based on our sine wave. We'll break it into 100
  // subintervals.
  for (int i = 1; i <= 20; ++i) {
    double input [] = {(4 * M_PI / 20) * i};
    double output [] = {sin(input[0])};
    learner.AddTrainingData(input, output);
  }

  EXPECT_TRUE(learner.Learn(0.03, 15000));
}

} // test
} // algorithm
