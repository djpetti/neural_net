// Tests for neuron class.

#include <vector>

#include "gtest/gtest.h"
#include "../neuron.h"
#include "../output_functions.h"

namespace network {
namespace testing {

TEST(NeuronTest, BasicTest) {
  // Test basic neuron functionality.
  Neuron neuron;
  Threshold threshold1(1);
  neuron.SetOutputFunction(&threshold1);
  
  std::vector<double> weights (3, 1);
  double input_array[] = {1, -1, 0};
  std::vector<double> inputs (input_array, input_array + sizeof(input_array) / sizeof(double));
  neuron.SetWeights(weights);
  neuron.SetInputs(inputs);

  double output;
  EXPECT_TRUE(neuron.GetOutput(&output));
  EXPECT_EQ(output, 0);
}

TEST(NeuronTest, WillFailTest) {
  // Tests whether failure conditions happen as they should.
  Neuron neuron;
  Threshold threshold1(1);
  neuron.SetOutputFunction(&threshold1);

  // Use a different number of weights and inputs.
  std::vector<double> weights (3, 1);
  std::vector<double> inputs (2, 1);
  neuron.SetWeights(weights);
  neuron.SetInputs(inputs);

  double output;
  EXPECT_FALSE(neuron.GetOutput(&output));
}

} //testing
} //network

