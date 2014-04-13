#define __STDC_FORMAT_MACROS
#include <inttypes.h> // temp
#include <stdio.h> // temp
#include <stdlib.h>
#include <time.h>

#include <algorithm>

#include "genetic_algorithm.h"

using ::network::Network;

namespace algorithm {

GeneticAlgorithm::GeneticAlgorithm(double crossover, double mutation) :
    generation_(0),
    total_fitness_(0),
    chromosome_size_(-1),
    crossover_rate_(crossover),
    mutation_rate_(mutation) {
      srand(time(NULL));
}

bool GeneticAlgorithm::AddNetwork(Network *network) {
  // Update the network's fitness score, to give the fitness function a chance
  // to properly initialize our network.
  int fitness = GetFitnessScore(network);
  if (chromosome_size_ != -1 &&
      network->GetChromosomeSize() != (size_t)chromosome_size_) {
    return false;
  }
  chromosome_size_ = network->GetChromosomeSize();
  if (!chromosome_size_) {
    return false;
  }
  networks_[network] = fitness;
  total_fitness_ += fitness;
  return true;
}

bool GeneticAlgorithm::RemoveNetwork(Network *network) {
  auto it = networks_.find(network);
  if (it == networks_.end()) {
    return false;
  }
  networks_.erase(it);
  
  return true;
}

void GeneticAlgorithm::NextGeneration() {
  // Pick and mate networks until we have an entirely new set of networks.
  uint64_t chromosomes [networks_.size()][chromosome_size_];

  for (uint32_t i = 0; i < networks_.size(); ++i) {
    Mate(PickRoulette(), PickRoulette(), chromosomes[i]);
  }

  // Since each generation is the same size, we can just reuse old networks.
  // Change our networks to their offspring.
  int index = 0;
  for (auto& kv : networks_) {
    kv.first->SetChromosome(chromosomes[index++]);
  }

  ++generation_;

  // Update our fitness scores for the new generation.
  UpdateFitness();
}

Network *GeneticAlgorithm::GetFittest() {
  uint32_t max = GetMaxFitness();
  for (auto& kv : networks_) {
    if (kv.second == max) {
      return kv.first;
    }
  } 
  return nullptr;
}

double GeneticAlgorithm::GetAverageFitness() {
  int sum = 0;
  for (auto& kv : networks_) {
    sum += kv.second;
  }
  return sum / networks_.size();
}

uint32_t GeneticAlgorithm::GetMaxFitness() {
  uint32_t max = 0;
  for (auto& kv : networks_) {
    max = std::max(kv.second, max);
  }
  return max;
}

void GeneticAlgorithm::UpdateFitness() {
  total_fitness_ = 0;
  for (auto& kv : networks_) {
    int fitness = GetFitnessScore(kv.first);
    while (fitness < 0) {
      // Make a new offspring to replace this one.
      uint64_t out_chromo [chromosome_size_];
      Mate(PickRoulette(), PickRoulette(), out_chromo);
      kv.first->SetChromosome(out_chromo);
      fitness = GetFitnessScore(kv.first);
    }
    total_fitness_ += fitness;
    networks_[kv.first] = fitness;
  }
}

Network *GeneticAlgorithm::PickRoulette() {
  int pick;
  if (!total_fitness_) {
    // Just pick a random one.
    pick = rand() % networks_.size() + 1;
    int traversed = 0;
    for (auto& kv : networks_) {
      if (++traversed >= pick) {
        return kv.first;
      }
    }
  } else {
    printf("Total fitness: %d\n", total_fitness_);
    pick = rand() % total_fitness_;
    printf("Pick: %d\n", pick);
  }
  int traversed = 0;
  for (auto& kv : networks_) {
    traversed += kv.second;
    if (traversed >= pick) {
     return kv.first; 
    }
  }
  ASSERT(false, "Something weird happened.");
}

void GeneticAlgorithm::Mate(Network *mother,
    Network *father, uint64_t *out_chromo) {
  // Extract individual chromosomes.
  printf("chromosome_size_: %d\n", chromosome_size_);
  uint64_t chromosome1 [chromosome_size_];
  uint64_t chromosome2 [chromosome_size_];
  const int type_len = sizeof(uint64_t) * 8;
  const uint64_t shifter = 1;
  mother->GetChromosome(chromosome1);
  father->GetChromosome(chromosome2);
  // Copy initial weights from our mother.
  mother->GetChromosome(out_chromo);

  // Handle recombination.
  int should_recombine = rand() % 100;
  printf("should_recombine: %d\n", should_recombine);
  int bitlen = chromosome_size_ * type_len;
  printf("bitlen: %d\n", bitlen);
  if (should_recombine < crossover_rate_ * 100) {
    // We need to recombine.
    // Pick a recombination threshold.
    int recombine_after = rand() % bitlen;
    bool first_word = true;
    printf("Got to start of recombination loop without crashing!\n");
    for (int i = recombine_after / type_len; 
        i < chromosome_size_;
        ++i) {
      if (first_word) {
        // Our first word might be a partial one.
        first_word = false;
        printf("Got to first word without crashing!\n");
        for (int bit = recombine_after % type_len; 
            bit < type_len; ++bit) {
          if (chromosome2[i] & (shifter << bit)) {
            // We need a one here.
            out_chromo[i] |= (shifter << bit);
          } else {
            // We need a zero here.
            out_chromo[i] &= ~(shifter << bit);
          }      
        }
      } else {
        // Now, we can just switch whole elements, which is faster and easier.
        out_chromo[i] = chromosome2[i];
      }
    }
  } else {
    // Just pick either the mother or the father's chromosome.
    // This is not an accurate representation of sexual reproduction.
    int choose_parent = rand() % 2 + 1;
    if (choose_parent == 1) {
      father->GetChromosome(out_chromo);
    }
    // We have the mother's chromosome by default.
  }

  // Handle mutation.
  for (int i = 0; i < bitlen; ++i) {
    int should_mutate = rand() % 1000 + 1;
    if (should_mutate <= mutation_rate_ * 1000) {
      out_chromo[i / type_len] ^= (shifter << (i % type_len));
    }
  }
}

} //algorithm
