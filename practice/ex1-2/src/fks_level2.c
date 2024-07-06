#include "../inc/fks_level2.h"
#include "../inc/hash_func.h"
#include <stddef.h>
#include <string.h>

fks_level2 *fks_level2_init(uint32_t size, hash_parameters parameters) {
  fks_level2 *level2;
  if ((level2 = malloc(sizeof(fks_level2))) == NULL) {
    return NULL;
  }
  if ((level2->slots = calloc(size, sizeof(uint32_t))) == NULL) {
    free(level2);
    return NULL;
  }
  for (size_t i = 0; i < size; i++) {
    level2->slots[i] = FKS_LEVEL2_EMPTY;
  }

  level2->size = size;
  level2->parameters = parameters;
  return level2;
}

fks_level2 *fks_level2_build(list_node *head, uint32_t size,
                             hash_parameters parameters) {
  fks_level2 *level2;
  if ((level2 = fks_level2_init(size, parameters)) == NULL) {
    return NULL;
  }

  list_node *node = head;
  while (node != NULL) {
    if (!fks_level2_insert(level2, node->key)) {
      fks_level2_destroy(level2);
      return NULL;
    }
    node = node->next;
  }

  level2->size = size;
  level2->parameters = parameters;
  return level2;
}

bool fks_level2_insert(fks_level2 *table, uint32_t key) {
  if (table != NULL) {
    uint32_t index = hash_func(key, table->parameters, table->size);
    if (table->slots[index] == FKS_LEVEL2_EMPTY) {
      table->slots[index] = key;
      return true;
    }
  }
  if (fks_level2_search(table, key)) {
    return true;
  }
  return false;
}

bool fks_level2_search(fks_level2 *table, uint32_t key) {
  if (table != NULL) {
    uint32_t index = hash_func(key, table->parameters, table->size);
    return table->slots[index] == key;
  }
  return false;
}

void fks_level2_destroy(fks_level2 *table) {
  if (table != NULL) {
    free(table->slots);
    free(table);
  }
}
