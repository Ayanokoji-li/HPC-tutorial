#ifndef HASH_PARAMETERS_H
#define HASH_PARAMETERS_H

#include <stdint.h>

typedef struct hash_parameters {
  uint32_t a;
  uint32_t b;
} hash_parameters;

/**
 * Generates hash parameters.
 *
 * @return The generated hash parameters.
 */
hash_parameters generate_hash_parameters(void);

#endif
