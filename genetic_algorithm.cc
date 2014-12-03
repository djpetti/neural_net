#include <inttypes.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>

#include "genetic_algorithm.h"
#include "logger.h"

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

bool GeneticAlgorithm::CheckNetwork(Network *network) {
  if (!network->GetChromosomeSize()) {
    return false;
  }
  if (chromosome_size_ != -1 &&
      network->GetChromosomeSize() != static_cast<size_t>(chromosome_size_)) {
    return false;
  }
  chromosome_size_ = network->GetChromosomeSize();
  if (!chromosome_size_) {
    return false;
  }
  return true;
}

bool GeneticAlgorithm::AddNetwork(Network *network) {
  // Update the network's fitness score, to give the fitness function a chance
  // to properly initialize our network.
  int fitness = GetFitnessScore(network);
  if (!CheckNetwork(network)) {
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

void GeneticAlgorithm::SortedFitnesses(::std::vector<uint32_t> & sorted) {
  for (auto & kv : networks_) {
    sorted.push_back(kv.second);
  }
  ::std::sort(sorted.begin(), sorted.end());
}

void GeneticAlgorithm::BuildHallOfFame(uint64_t **chromosomes) {
  hall_of_famers_.clear();

  ::std::vector<uint32_t> fitnesses;
  SortedFitnesses(fitnesses);
  // Truncate to the ones that will be in our hall of fame.
  fitnesses.erase(fitnesses.begin(), fitnesses.end() - hall_of_fame_size_);
  CHECK(fitnesses.size() == hall_of_fame_size_,
      "Did not get correctly sized fitness list.");

  // Find the networks that go with these fitnesses.
  int added = 0;
  int number_expected = 1;
  for (uint32_t i = 0; i < fitnesses.size(); ++i) {
    LOG(Level::DEBUG, "Hall of fame fitness: %" PRIu32 ".", fitnesses[i]);
    if (i < fitnesses.size() - 1 && fitnesses[i] == fitnesses[i + 1]) {
      ++number_expected;
      continue;
    }

    int found = 0;
    for (auto & kv : networks_) {
      if (kv.second == fitnesses[i]) {
        hall_of_famers_.push_back(kv.first);
        kv.first->GetChromosome(chromosomes[added++]);
        if (++found == number_expected) {
          break;
        }
      }
    }
    CHECK(found == number_expected,
          "Did not find expected number of networks.");
    number_expected = 1;
  }
  CHECK(added == static_cast<int>(hall_of_fame_size_),
        "Did not put the right number of networks in hall of fame.");
}

void GeneticAlgorithm::NextGeneration() {
  // Pick and mate networks until we have an entirely new set of networks.
  if (networks_.empty()) {
    return;
  }

  // This array can get really huge, and that's why we need to put it on the
  // heap.
  uint64_t **chromosomes = new uint64_t *[networks_.size()];
  for (uint32_t i = 0; i < networks_.size(); ++i) {
    chromosomes[i] = new uint64_t[chromosome_size_];
  }

  // Incorporate hall of fame organisms into the population.
  BuildHallOfFame(chromosomes);
  for (uint32_t i = hall_of_fame_size_; i < networks_.size(); ++i) {
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

  for (uint32_t i = 0; i < networks_.size(); ++i) {
    delete[] chromosomes[i];
  }
  delete[] chromosomes;
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
  double average = total_fitness_ / networks_.size();
  return average;
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
  for (auto & kv : networks_) {
    if (::std::find(hall_of_famers_.begin(), hall_of_famers_.end(), kv.first) !=
        hall_of_famers_.end()) {
      // We already know the fitness for that one.
      continue;
    }

    int fitness = GetFitnessScore(kv.first);
    while (fitness < 0) {
      // Make a new offspring to replace this one.
      uint64_t out_chromo [chromosome_size_];
      Mate(PickRoulette(), PickRoulette(), out_chromo);
      kv.first->SetChromosome(out_chromo);
      fitness = GetFitnessScore(kv.first);
    }
    total_fitness_ += fitness;
    kv.second = fitness;
  }
}

Network *GeneticAlgorithm::PickRoulette() {
  int pick;
  if (!total_fitness_) {
    // Just pick a random one.
    pick = rand() % networks_.size() + 1;
    int traversed = 0;
    for (auto & kv : networks_) {
      if (++traversed >= pick) {
        return kv.first;
      }
    }
  }

  ::std::vector<uint32_t> fitnesses;
  SortedFitnesses(fitnesses);
  // Remove duplicates.
  int total = fitnesses[0];
  for (uint32_t i = 1; i < fitnesses.size(); ++i) {
    if (fitnesses[i] == fitnesses[i - 1]) {
      fitnesses.erase(fitnesses.begin() + i);
      --i;
    } else {
      total += fitnesses[i];
    }
  }

  pick = rand() % total;

  int traversed = 0;
  for (auto fitness : fitnesses) {
    traversed += fitness;
    if (traversed >= pick) {
      for (auto & kv : networks_) {
        if (kv.second == fitness) {
          return kv.first;
        }
      }
    }
  }
  LOG(Level::FATAL, "Something weird happened.");
  return nullptr;
}

void GeneticAlgorithm::Mate(Network *mother,
    Network *father, uint64_t *out_chromo) {
  // Extract individual chromosomes.
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
  int bitlen = chromosome_size_ * type_len;
  if (should_recombine < crossover_rate_ * 100) {
    // We need to recombine.
    // Pick a recombination threshold.
    int recombine_after = rand() % bitlen;
    bool first_word = true;
    for (int i = recombine_after / type_len;
        i < chromosome_size_;
        ++i) {
      if (first_word) {
        // Our first word might be a partial one.
        first_word = false;
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
