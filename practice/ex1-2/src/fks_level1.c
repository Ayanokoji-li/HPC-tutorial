#include "../inc/fks_level1.h"
#include "../inc/hash_func.h"
#include <stdint.h>
#include <stdio.h>

static int get_list_len(list_node *node) {
  int ans = 0;
  while (node != NULL) {
    ans++;
    node = node->next;
  }
  return ans;
}

fks_level1 *fks_level1_build(hash_chaining *hash_chaining_table) {
  fks_level1 *level1;
  if ((level1 = malloc(sizeof(fks_level1))) == NULL) {
    return NULL;
  }
  if ((level1->level2_tables =
           calloc(hash_chaining_table->size, sizeof(fks_level2 *))) == NULL) {
    free(level1);
    return NULL;
  }

  for (size_t i = 0; i < hash_chaining_table->size; i++) {
    struct list_node *node = hash_chaining_table->slots[i];
    if (node != NULL) {
      uint32_t len = get_list_len(node);
      while (1) {
        if ((level1->level2_tables[i] = fks_level2_build(
                 node, len * len, generate_hash_parameters())) != NULL) {
          break;
        }
      }
    } else {
      level1->level2_tables[i] = NULL;
    }
  }

  level1->size = hash_chaining_table->size;
  level1->parameters = hash_chaining_table->parameters;
  return level1;
}

bool fks_level1_search(fks_level1 *table, uint32_t key) {
  if (table != NULL) {
    uint32_t index = hash_func(key, table->parameters, table->size);
    fks_level2 *level2 = table->level2_tables[index];
    if (level2 != NULL) {
      return fks_level2_search(level2, key);
    }
  }
  return false;
}

void fks_level1_destroy(fks_level1 *table) {
  if (table != NULL) {
    for (size_t i = 0; i < table->size; i++) {
      fks_level2 *level2 = table->level2_tables[i];
      if (level2 != NULL) {
        fks_level2_destroy(level2);
      }
    }
    free(table->level2_tables);
    free(table);
  }
}
