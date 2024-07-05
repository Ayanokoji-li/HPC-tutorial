#ifndef FKS_LEVEL2_H
#define FKS_LEVEL2_H

#include "hash_chaining.h"
#include "hash_parameters.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define FKS_LEVEL2_EMPTY UINT32_MAX

typedef struct fks_level2 {
  uint32_t size;
  uint32_t *slots;
  hash_parameters parameters;
} fks_level2;

/**
 * Initializes a fks_level2 instance with the specified size and hash
 * parameters. The slots should be initialized to FKS_LEVEL2_EMPTY.
 * @param size The size of the fks_level2 instance.
 * @param parameters The hash parameters of fks_level2 instance.
 * @return A pointer to the initialized fks_level2 instance.
 */
fks_level2 *fks_level2_init(uint32_t size, hash_parameters parameters);

/**
 * Initialize a fks_level2 instance with specified size and put all elements in
 * the linked list into fks_level2 instance. If there are collisions, free
 * allocated resources and return NULL.
 *
 * @param head The head of the linked list containing the elements.
 * @param size The size of fks_level2 instance.
 * @param parameters The hash parameters of fks_level2 instance.
 * @return If there are not collisions, a pointer to the fks_level2 instance is
 * returned. Otherwise, return NULL.
 */
fks_level2 *fks_level2_build(list_node *head, uint32_t size,
                             hash_parameters parameters);

/**
 * Try to insert a key into the fks_level2 table.
 *
 * @param table The fks_level2 table to insert the key into.
 * @param key The key to be inserted.
 * @return true if the key was successfully inserted (no collision), false
 * otherwise.
 */
bool fks_level2_insert(fks_level2 *table, uint32_t key);

/**
 * Searches for a key in the fks_level2 table.
 *
 * @param table The fks_level2 table to search in.
 * @param key The key to search for.
 * @return true if the key is found, false otherwise.
 */
bool fks_level2_search(fks_level2 *table, uint32_t key);

/**
 * @brief Destroys a fks_level2 table.
 *
 * This function frees the memory allocated for a fks_level2 table.
 *
 * @param table Pointer to the fks_level2 table to be destroyed.
 */
void fks_level2_destroy(fks_level2 *table);

#endif
