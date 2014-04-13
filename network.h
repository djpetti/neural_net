#ifndef NEURAL_NET_NETWORK_H_
#define NEURAL_NET_NETWORK_H_

#include <stdint.h>
#include <string.h>

#include "macros.h"

// A superclass for all neural networks.

namespace network {

class Network {
 public:
  // TODO(danielp): Investigate: a) Why it doesn't automatically generate a
  // default constructor and b) Why it will happily use a copy constructor
  // instead, if available.
  Network() = default;
  virtual ~Network() {};
  // These methods are necessary for genetic algorithms to work right.
  // This is for extracting usable chromosome data from the network.
  // <chromosome> is the output.
  virtual bool GetChromosome(uint64_t *chromosome) = 0;
  // We also care about the size of the chromosome.
  virtual size_t GetChromosomeSize() = 0;
  // Sets the chromosome. Note that the size of the array passed in must be at
  // least equal to whatever GetChromosomeSize() returns.
  virtual bool SetChromosome(uint64_t *chromosome) = 0;

  DISSALOW_COPY_AND_ASSIGN(Network);
};

} //network

#endif
