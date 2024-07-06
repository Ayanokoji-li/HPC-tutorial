#ifndef HASH_FUNC_H
#define HASH_FUNC_H
#include "hash_parameters.h"
#include <stdint.h>

/**
 * Computes the hash value for a given key using the specified hash parameters
 * and hash table size.
 *
 * @param key The key value.
 * @param parameters The hash parameters used for computing the hash value.
 * @param m The size of the hash table.
 * @return The computed hash value.
 */
uint32_t hash_func(uint32_t key, hash_parameters parameters, uint32_t m);

#endif
