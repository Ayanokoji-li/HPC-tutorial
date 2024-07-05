#ifndef HASH_CHAINING_H
#define HASH_CHAINING_H

#include "hash_parameters.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct hash_chaining {
  uint32_t size;
  struct list_node **slots;
  hash_parameters parameters;
} hash_chaining;

typedef struct list_node {
  uint32_t key;
  struct list_node *next;
} list_node;

/**
 * Initializes a hash table using chaining collision resolution.
 *
 * @param size The size of the hash table.
 * @return A pointer to the initialized hash table.
 */
hash_chaining *hash_chaining_init(uint32_t size);

/**
 * Inserts a key into the hash table using chaining collision resolution.
 *
 * @param table The hash table to insert the key into.
 * @param key The key to be inserted.
 */
void hash_chaining_insert(hash_chaining *table, uint32_t key);

/**
 * Searches for a key in the hash table using chaining.
 *
 * @param table The hash table to search in.
 * @param key The key to search for.
 * @return true if the key is found, false otherwise.
 */
bool hash_chaining_search(hash_chaining *table, uint32_t key);

/**
 * Destroys the hash table. Frees the memory allocated for the hash table and
 * all its elements.
 *
 * @param table A pointer to the hash table to be destroyed.
 */
void hash_chaining_destroy(hash_chaining *table);

#endif
