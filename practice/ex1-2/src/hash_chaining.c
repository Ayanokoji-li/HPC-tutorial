#include "../inc/hash_chaining.h"
#include "../inc/hash_func.h"
#include "../inc/hash_parameters.h"

hash_chaining *hash_chaining_init(uint32_t size) {
  hash_chaining *hash;
  if ((hash = malloc(sizeof(hash_chaining))) == NULL) {
    return NULL;
  }
  if ((hash->slots = calloc(size, sizeof(struct list_node *))) == NULL) {
    free(hash);
    return NULL;
  }

  hash->size = size;
  hash->parameters = generate_hash_parameters();
  return hash;
}

void hash_chaining_insert(hash_chaining *table, uint32_t key) {
  if (table != NULL) {
    uint32_t index = hash_func(key, table->parameters, table->size);
    struct list_node *node;
    if ((node = malloc(sizeof(struct list_node))) == NULL) {
      return;
    }

    node->key = key;
    node->next = table->slots[index];
    table->slots[index] = node;
  }
}

bool hash_chaining_search(hash_chaining *table, uint32_t key) {
  if (table != NULL) {
    uint32_t index = hash_func(key, table->parameters, table->size);
    struct list_node *node = table->slots[index];
    while (node != NULL) {
      if (node->key == key) {
        return true;
      }
      node = node->next;
    }
  }
  return false;
}

void hash_chaining_destroy(hash_chaining *table) {
  if (table != NULL) {
    for (size_t i = 0; i < table->size; i++) {
      struct list_node *node = table->slots[i];
      while (node != NULL) {
        struct list_node *next = node->next;
        free(node);
        node = next;
      }
    }
    free(table->slots);
    free(table);
  }
}
