// Tests for multilayered feedforward network.

#include <stdio.h>
#include <stdint.h>

#include "gtest/gtest.h"
#include "../multilayered_feedforward.h"
#include "../output_functions.h"

namespace network {
namespace testing {

TEST(BasicTest, FullTest) {
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
  // It should return false, since we haven't set an output function.
  EXPECT_FALSE(network.GetOutputs(values));
  network.SetOutputFunctions(&threshold1);
  // NOW it should return true...
  EXPECT_TRUE(network.GetOutputs(values));
  printf("Got output: %f\n", values[0]);
}

TEST(BasicTest, XorTest) {
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
  EXPECT_TRUE(network.SetLayerWeights(0, weights));
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

TEST(BasicTest, FileIOTest) {
  // Can we save and load networks to and from files?
  MFNetwork network (1, 1, 1);
  network.AddHiddenLayer();
  
  size_t size = network.GetChromosomeSize();
  uint64_t chromosome1 [size];
  uint64_t chromosome2 [size];
  network.GetChromosome(chromosome1);
  EXPECT_TRUE(MFNetwork::SaveToFile("test.bin", &network));
  EXPECT_TRUE(MFNetwork::ReadFromFile("test.bin", &network));
  network.GetChromosome(chromosome2);
  
  for (uint32_t i = 0; i < size; ++i) {
    EXPECT_EQ(chromosome1[i], chromosome2[i]);
  }
}

TEST(GenAlgTest, ChromosomeMethodsTest) {
  // Test whether we can get and set chromosomes correctly.
  MFNetwork network (1, 1, 2);
  network.AddHiddenLayer();

  int size = network.GetChromosomeSize();
  uint64_t chromosome [size];
  ASSERT_TRUE(network.GetChromosome(chromosome));
 
  for (int i = 0; i < size; ++i) {
    // The default weight is 1, so that's what it should be.
    EXPECT_EQ(1, chromosome[i]);
  }

  // Write a changed chromosome.
  chromosome[0] = 2;
  ASSERT_TRUE(network.SetChromosome(chromosome));
  // See if this is now reflected.
  ASSERT_TRUE(network.GetChromosome(chromosome));
  EXPECT_EQ(2, chromosome[0]);
  for (int i = 1; i < size; ++i) {
    EXPECT_EQ(1, chromosome[i]);
  }
}

} //testing
} //network
