// Defines some macros that I basically need everywhere.

#ifndef NEURAL_NET_MACROS_H_
#define NEURAL_NET_MACROS_H_

#include <stdio.h>
#include <stdlib.h>

// This requires c++ '11.
#define DISSALOW_COPY_AND_ASSIGN(T) \
    T & operator=(const T&) = delete; \
    T(const T&) = delete;

// I don't like C asserts that much...
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            fprintf(stderr, "Assertion `" #condition "` failed in:\
            %s line %d: " #message "\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

#endif
