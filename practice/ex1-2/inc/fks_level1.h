#ifndef FKS_LEVEL1_H
#define FKS_LEVEL1_H

#include "fks_level2.h"
#include "hash_chaining.h"
#include "hash_parameters.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct fks_level1 {
  uint32_t size;
  fks_level2 **level2_tables;
  hash_parameters parameters;
} fks_level1;

/**
 * Builds a fks_level1 object using the provided hash_chaining table.
 *
 * @param hash_chaining_table The hash_chaining table to be used for building
 * the fks_level1 object.
 * @return A pointer to the newly created fks_level1 instance.
 */
fks_level1 *fks_level1_build(hash_chaining *hash_chaining_table);

/**
 * Searches for a given key in the fks_level1 table.
 *
 * @param table A pointer to the fks_level1 table.
 * @param key The key to search for.
 * @return true if the key is found in the table, false otherwise.
 */
bool fks_level1_search(fks_level1 *table, uint32_t key);

/**
 * @brief Destroys the fks_level1 table.
 *
 * This function is responsible for destroying the fks_level1 table and freeing
 * up any allocated memory.
 *
 * @param table A pointer to the fks_level1 table to be destroyed.
 */
void fks_level1_destroy(fks_level1 *table);

#endif
