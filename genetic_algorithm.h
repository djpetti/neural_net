#ifndef NEURAL_NET_GENETIC_ALGORITHM_H_
#define NEURAL_NET_GENETIC_ALGORITHM_H_

// A simple class for implementing genetic algorithms, designed to work with
// various forms of neural networks.
// NOTE: At this point, all the individual networks in the population must have
// the same layout. If they don't, the code should still run, but it WILL NOT
// BEHAVE AS YOU EXPECT IT TO!

// Some important things about networks in the population: The genetic
// algorithm instance DOES NOT take ownership of any of the networks in its
// population. You are fully allowed to use them and change them while they are
// in the population, but you are responsible for freeing them, if they are on
// the heap. Also, the algorithm uses the pointers to networks that you pass in
// internally, (there is no copying), so don't do any dumb things with stack
// pointers that dissapear out from under it and such.

#include <stdint.h>

#include <map>
#include <vector>

#include "macros.h"
#include "network.h"
#include "string.h"

namespace algorithm {

class GeneticAlgorithm {
 public:
  // The ctor lets you specify crossover and mutation rates.
  GeneticAlgorithm(double crossover, double mutation);
  // Adds a new network to the algorithm's population.
  // The chromosome size is set by the first one added, adding
  // different sized ones will cause it to return false.
  bool AddNetwork(network::Network *network);
  // Removes a network from the algorithm's population.
  // It returns false if it can't find the requested network.
  bool RemoveNetwork(network::Network *network);
  // Computes one generation in the algorithm.
  void NextGeneration();
  // Obtains a pointer to the fittest individual, which can be NULL if the
  // population is empty.
  network::Network *GetFittest();
  // Gets the average fitness of the population.
  double GetAverageFitness();
  // Gets the best fitness.
  uint32_t GetMaxFitness();
  // Gets the population size.
  inline size_t GetPopulationSize() {
    return networks_.size();
  }
  // Gets the current generation number.
  inline uint32_t GetGeneration() {
    return generation_;
  }
  // Specifies the hall of fame size.
  inline void SetHallOfFameSize(uint32_t size) {
    hall_of_fame_size_ = size;
  }

  DISSALOW_COPY_AND_ASSIGN(GeneticAlgorithm);

 protected:
  // Instead of function pointers, I prefer to let the user implement their own
  // fitness function by subclassing, since it shouldn't change during the
  // lifetime of the class, and there are no truly standard presets which
  // I can offer. It must take a pointer to the network you are evaluating.
  virtual int GetFitnessScore(network::Network *network) = 0;
  // Checks that it is okay to add a network to our population.
  // network: The network being checked.
  bool CheckNetwork(::network::Network *network);

  // Maps each member of the population to its fitness score. Some subclasses
  // need this.
  std::map<network::Network *, uint32_t> networks_;

 private:
  // Goes through all the networks in the population and recalculates fitness
  // scores for them.
  void UpdateFitness();
  // Returns either a fitness or -1 if the organism is not viable.
  int GetViableFitness(network::Network *network);
  // Goes and gets a network by simulated roulette.
  network::Network *PickRoulette();
  // Takes two networks, and sets out_chromo to represent a
  // combined, mutated offspring.
  void Mate(network::Network *mother, network::Network *father,
      uint64_t *out_chromo);
  // Puts members of the hall of fame in the next generation. <chromosomes> is
  // the chromosomes array for the next generation.
  void BuildHallOfFame(uint64_t **chromosomes);
  // Populates a vector of sorted fitnesses.
  void SortedFitnesses(::std::vector<uint32_t> & sorted);

  // A vector of pointers to all the networks in our hall of fame.
  ::std::vector<::network::Network *> hall_of_famers_;
  // A counter of our current generation.
  uint32_t generation_;
  // A measure of the total fitness of all networks.
  uint32_t total_fitness_;
  // The size of the hall of fame.
  uint32_t hall_of_fame_size_ = 0;
  // How many uint64_t's are in each chromosome.
  int chromosome_size_;
  // The crossover rate for all chromosomes.
  double crossover_rate_;
  // The mutation rate for all chromosomes.
  double mutation_rate_;
};

} //algorithm

#endif
