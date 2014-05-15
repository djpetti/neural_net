// Defines some macros that I basically need everywhere.

#ifndef NEURAL_NET_MACROS_H_
#define NEURAL_NET_MACROS_H_

#include <stdio.h>
#include <stdlib.h>

// This requires c++ '11.
#define DISSALOW_COPY_AND_ASSIGN(T) \
    T & operator=(const T&) = delete; \
    T(const T&) = delete;

#endif
