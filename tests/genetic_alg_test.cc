// Tests for genetic algorithm class.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <vector>

#include "../genetic_algorithm.h"
#include "../multilayered_feedforward.h"
#include "../network.h"
#include "../output_functions.h"
#include "gtest/gtest.h"

using network::MFNetwork;
using network::Network;

namespace algorithm {
namespace test {

static_assert(sizeof(double) == sizeof(uint64_t), 
    "Tests will fail randomly otherwise.");

// Genetic algorithm forces us to subclass it and implement a fitness function.
class TestAlg : public GeneticAlgorithm {
 public:
  // <fitness_ret> allows one to specify a fitness value to return.
  TestAlg(double crossover, double mutation, int fitness_ret = -1) :
      GeneticAlgorithm(crossover, mutation),
      fitness_ret_(fitness_ret),
      roulette_pos_(0) {}

  virtual int GetFitnessScore(Network *network) {
    MFNetwork *mfnetwork = dynamic_cast<MFNetwork *>(network);
    double inputs[] = {1};
    mfnetwork->SetInputs(inputs);
    double outputs [1];
    mfnetwork->GetOutputs(outputs);
    printf("Outputs: %f\n", outputs[0]);
    return fitness_ret_ == -1 ? abs(floor(outputs[0])) : fitness_ret_;
  }

 private:
  // Picks things in a predictable fashion instead of randomly.
  Network *PickRoulette() {
    uint32_t traversed = 0;
    for (auto& kv : networks_) {
      if (traversed++ >= roulette_pos_) {
        ++roulette_pos_;
        if (roulette_pos_ == networks_.size()) {
          roulette_pos_ = 0;
        }
        return kv.first;
      }
    }
  }

  int fitness_ret_;
  uint32_t roulette_pos_;
};

class HundredGA : public GeneticAlgorithm {
 public:
  HundredGA() : GeneticAlgorithm(0.5, 0.006) {}

  virtual int GetFitnessScore(Network *network) {
    MFNetwork *mfnetwork = dynamic_cast<MFNetwork *>(network);
    double inputs[] = {1};
    mfnetwork->SetInputs(inputs);
    double outputs [1];
    mfnetwork->GetOutputs(outputs);
    if (isnan(outputs[0])) {
      return -1;
    }
    int error = abs(100 - floor(outputs[0]));
    int fitness = 100 - error;
    return std::max(fitness, 0);
  }
};

// Superclass for basic algorithm testing fixture.
class GABasicTest : public ::testing::Test {
 public:
  GABasicTest() :
      network_(1, 1, 1),
      alg_(0, 0),
      threshold0_(0) {}

 protected:
  MFNetwork network_;
  TestAlg alg_;
  network::Threshold threshold0_;

  virtual void SetUp() {
    network_.AddHiddenLayer();
    network_.RandomWeights(-10, 10);
    network_.SetOutputFunctions(&threshold0_);
    ASSERT_TRUE(alg_.AddNetwork(&network_));
  }
};

TEST_F(GABasicTest, AddRemoveTest) {
  // Can we add and remove networks in the population?
  EXPECT_TRUE(alg_.RemoveNetwork(&network_));
  // If we already removed it, it shouldn't work.
  EXPECT_FALSE(alg_.RemoveNetwork(&network_));
  // Add it back again.
  EXPECT_TRUE(alg_.AddNetwork(&network_));
}

TEST_F(GABasicTest, FitnessTest) {
  // How well do the fitness semantics work?
  alg_.NextGeneration();
  // Check for fitness data.
  MFNetwork *fittest = dynamic_cast<MFNetwork *>(alg_.GetFittest());
  EXPECT_EQ(fittest, &network_);
}

TEST_F(GABasicTest, MutationTest) {
  // Can we successfully mutate chromosomes?
  size_t size = network_.GetChromosomeSize();
  uint64_t initial [size];
  network_.GetChromosome(initial);

  TestAlg alg (0, 1);
  ASSERT_TRUE(alg.AddNetwork(&network_));
  alg.NextGeneration();

  // Since mutation rate was one, we should have flipped all the bits.
  uint64_t final [size];
  network_.GetChromosome(final);
  for (uint32_t i = 0; i < size; ++i) {
    EXPECT_EQ(final[i], ~(initial[i]));
  }
}

TEST(GATest, RecombinationTest) {
  // Can we recombine chromosomes?
  MFNetwork network (1, 1, 1);
  MFNetwork network2 (1, 1, 1);
  network.AddHiddenLayer();
  network2.AddHiddenLayer();
  uint64_t all_ones_i = 1;
  all_ones_i <<= 63;
  --all_ones_i;
  const uint64_t all_zeros_i = 0;
  double all_ones;
  double all_zeros;
  memcpy(&all_ones, &all_ones_i, sizeof(uint64_t));
  memcpy(&all_zeros, &all_ones_i, sizeof(uint64_t));
  network.SetWeights(all_ones);
  network2.SetWeights(all_zeros);
  network.SetBiases(all_ones);
  network2.SetBiases(all_zeros);
  network::Threshold threshold0(0);
  network.SetOutputFunctions(&threshold0);
  network2.SetOutputFunctions(&threshold0);
  TestAlg alg (1, 0, 1);
  ASSERT_TRUE(alg.AddNetwork(&network));
  ASSERT_TRUE(alg.AddNetwork(&network2));

  alg.NextGeneration();

  // Now, our recombined genome should be easily detectable.
  int switched = 0;
  int switched2 = 0;
  bool last_current = false;
  bool last_current2 = false;
  bool current = false;
  bool current2 = false;
  size_t size = network.GetChromosomeSize();
  uint64_t chromosome [size];
  uint64_t chromosome2 [size];
  network.GetChromosome(chromosome);
  network2.GetChromosome(chromosome2);
  for (uint32_t chromo_i = 0; chromo_i < size; ++chromo_i) {
    for (uint32_t gene_i = 0; gene_i < sizeof(uint64_t) * 8; ++gene_i) {
      current = chromosome[chromo_i] & (1 << gene_i);
      current2 = chromosome2[chromo_i] & (1 << gene_i);
      if (current != last_current) {
        // It changed here. We should only have one switch for the entire
        // chromosome, but, depending on how it stores arrays in memory, a
        // second switch at the beginning of a gene is also valid.
        if (switched && !(switched == 1 && gene_i == 0)) {
          // Invalid.
          ADD_FAILURE() << "First switched " << switched << " times.";
        }
        ++switched;
      }
      if (current2 != last_current2) {
        if (switched2 && !(switched2 == 1 && gene_i == 0)) {
          ADD_FAILURE() << "Second switched " << switched2 << " times.";
        }
        ++switched2;
      }

      last_current = current;
      last_current2 = current2;
    }
  }
}

TEST(GATest, HundredOutputTest) {
  // Tries to evolve a network that outputs 100 after inputting 1.
  std::vector<MFNetwork *> networks;
  for (int i = 0; i < 100; ++i) {
    networks.push_back(new MFNetwork(1, 1, 1));
  }

  HundredGA alg;
  network::DumbOutputer dumboutputer;
  for (int i = 0; i < 100; ++i) {
    networks[i]->AddHiddenLayer();
    networks[i]->AddHiddenLayer();
    networks[i]->AddHiddenLayer();
    networks[i]->RandomWeights(-50, 50);
    networks[i]->SetOutputFunctions(&dumboutputer);
    ASSERT_TRUE(alg.AddNetwork(networks[i]));
  }

  while (alg.GetMaxFitness() != 100) {
    alg.NextGeneration();
    printf("Generation: %d\n", alg.GetGeneration());
    printf("Max Fitness: %d\n", alg.GetMaxFitness());
    printf("Average Fitness: %f\n", alg.GetAverageFitness());
    sleep(1);
  } 

  for (MFNetwork *network : networks) {
    delete network;
  }
}

} //test
} //algorithm
