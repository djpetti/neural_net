// Tests for multilayered feedforward network.

#include <stdint.h>
#include <string.h>

#include "gtest/gtest.h"
#include "../logger.h"
#include "../multilayered_feedforward.h"
#include "../output_functions.h"

namespace network {
namespace testing {

TEST(BasicTests, FullTest) {
  // Basically a one-pass test for everything.
  MFNetwork network (1, 1, 2);
  
  // Add two hidden layers.
  network.AddHiddenLayer();
  EXPECT_EQ(network.HiddenLayerQuantity(), 1);
  network.AddHiddenLayer();
  EXPECT_EQ(network.HiddenLayerQuantity(), 2);

  // Initialize all the weights randomly.
  network.RandomWeights(-5, 10);

  // This is not a particularly useful test, it's mostly there to make sure we
  // can write some inputs and get some outputs without it segfaulting.
  double values [] = {10};
  Threshold threshold1(1);
  network.SetInputs(values);
  EXPECT_TRUE(network.GetOutputs(values));
  network.SetOutputFunctions(&threshold1);
  // NOW it should return true...
  EXPECT_TRUE(network.GetOutputs(values));
  printf("Got output: %f\n", values[0]);
}

TEST(BasicTests, XorTest) {
  // Create a basic xoring network and see if it performs as expected.
  MFNetwork network (2, 1, 3);
  network.AddHiddenLayer();
  // We set output functions in stages.
  Threshold threshold1(1);
  Threshold threshold2(2);
  DumbOutputer dumboutputer; 
  network.SetOutputFunctions(&dumboutputer);
  EXPECT_TRUE(network.SetLayerOutputFunctions(1, &threshold1));
  network.GetNeuron(1, 1)->SetOutputFunction(&threshold2);
  // Set up weights.
  std::vector<double> out_weights (3, 1);
  std::vector<double> weights (1, 1);
  out_weights[1] = -2;
  EXPECT_TRUE(network.SetLayerWeights(1, weights));
  EXPECT_TRUE(network.SetLayerWeights(2, out_weights));
  weights.push_back(1);
  network.GetNeuron(1, 1)->SetWeights(weights);
  // Set up routes.
  std::vector<int> routes (2, 0);
  routes[1] = 1;
  network.SetOutputRoute(0, 0, routes);
  routes[0] = 1;
  routes[1] = 2;
  network.SetOutputRoute(0, 1, routes);

  // See if it works.
  double values [] = {1, 0};
  double out [1];
  network.SetInputs(values);
  EXPECT_TRUE(network.GetOutputs(out));
  EXPECT_EQ(out[0], 1);
  // 0, 1
  values[0] = 0;
  values[1] = 1;
  network.SetInputs(values);
  EXPECT_TRUE(network.GetOutputs(out));
  EXPECT_EQ(out[0], 1);
  // 0, 0
  values[1] = 0;
  network.SetInputs(values);
  EXPECT_TRUE(network.GetOutputs(out));
  EXPECT_EQ(out[0], 0);
  // 1, 1
  values[0] = 1;
  values[1] = 1;
  network.SetInputs(values);
  EXPECT_TRUE(network.GetOutputs(out));
  EXPECT_EQ(out[0], 0);
}

TEST(BasicTests, FileIOTest) {
  // Can we save and load networks to and from files?
  MFNetwork network(1, 1, 1);
  network.AddHiddenLayer();
  
  size_t chromo_size = network.GetChromosomeSize();
  uint64_t chromosome1 [chromo_size];
  uint64_t chromosome2 [chromo_size];
  network.GetChromosome(chromosome1);
  EXPECT_TRUE(network.SaveToFile("test.bin"));

  MFNetwork network2(2, 2, 2);
  EXPECT_TRUE(network2.ReadFromFile("test.bin"));
  network2.GetChromosome(chromosome2);
  
  EXPECT_EQ(network.HiddenLayerQuantity(), network2.HiddenLayerQuantity());
  for (uint32_t i = 0; i < chromo_size; ++i) {
    EXPECT_EQ(chromosome1[i], chromosome2[i]);
  }
}

TEST(BackPropagationTests, DecreasingErrorTest) {
  // Does the error actually decrease when we run the back propagation
  // algorithm?
  helpers::Logger::Show(Level::INFO);

  MFNetwork network(1, 1, 5);
  network.AddHiddenLayer();
  network.AddHiddenLayer();
  network.RandomWeights(-1, 1);
  Sigmoid sigmoid;
  network.SetOutputFunctions(&sigmoid);
 
  double input = 0.01;
  network.SetInputs(&input);
  
  double initial_output;
  ASSERT_TRUE(network.GetOutputs(&initial_output));
  double initial_error = fabs(0.5 - initial_output);
  double target = 0.5;
  for (int i = 0; i < 100; ++i) {
    network.PropagateError(&target);
  }
  double final_output;
  ASSERT_TRUE(network.GetOutputs(&final_output));
  double final_error = fabs(0.5 - final_output);
  LOG(Level::INFO, "Initial error: %.10f", initial_error);
  LOG(Level::INFO, "Final error: %.10f", final_error);
  EXPECT_LE(final_error, initial_error);
}

TEST(GenAlgTest, ChromosomeMethodsTest) {
  // Test whether we can get and set chromosomes correctly.
  MFNetwork network (1, 1, 2);
  network.AddHiddenLayer();
  network.SetWeights(1);
  network.SetBiases(1);

  int size = network.GetChromosomeSize();
  uint64_t chromosome [size];
  ASSERT_TRUE(network.GetChromosome(chromosome));

  // This is what evetrything in our chromosome should be equal to.
  double double_one = 1;
  uint64_t chromo_value;
  memcpy(&chromo_value, &double_one, sizeof(chromo_value));
 
  for (int i = 0; i < size; ++i) {
    // The default weight is 1, so that's what it should be.
    EXPECT_EQ(chromo_value, chromosome[i]);
  }

  // Write a changed chromosome.
  chromosome[0] = 2;
  ASSERT_TRUE(network.SetChromosome(chromosome));
  // See if this is now reflected.
  ASSERT_TRUE(network.GetChromosome(chromosome));
  EXPECT_EQ(2, chromosome[0]);
  for (int i = 1; i < size; ++i) {
    EXPECT_EQ(chromo_value, chromosome[i]);
  }
}

} //testing
} //network
