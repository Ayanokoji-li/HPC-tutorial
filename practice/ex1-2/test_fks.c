#include "inc/fks_level1.h"
#include "inc/hash_chaining.h"
#include "inc/hash_func.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXN 100000

uint32_t inserted_cnt = MAXN;
uint32_t inserted[MAXN];

uint32_t in_chaining_cnt = 0;
uint32_t in_chaining[MAXN];

uint32_t in_fks_cnt = 0;
uint32_t in_fks[MAXN];

int cmp(const void *a, const void *b) {
  return (*(uint32_t *)a) < (*(uint32_t *)b);
}

// adapted from: https://en.cppreference.com/w/cpp/algorithm/unique
uint32_t unique(uint32_t a[], uint32_t a_n) {
  uint32_t first = 0, last = a_n;
  uint32_t result = 0;
  if (first == last)
    return last;
  while (++first < last) {
    if (a[result] != a[first] && ++result != first) {
      a[result] = a[first];
    }
  }
  return ++result;
}

void check_set_equal(uint32_t a[], uint32_t a_n, uint32_t b[], uint32_t b_n) {
  qsort(a, a_n, sizeof(uint32_t), cmp);
  qsort(b, b_n, sizeof(uint32_t), cmp);

  a_n = unique(a, a_n);
  b_n = unique(b, b_n);

  assert(a_n == b_n);

  for (uint32_t i = 0; i < a_n; ++i) {
    assert(a[i] == b[i]);
  }
}

void check_chaining_property(hash_chaining *chaining) {
  for (uint32_t i = 0; i < chaining->size; ++i) {
    list_node *node = chaining->slots[i];
    while (node) {
      assert(hash_func(node->key, chaining->parameters, chaining->size) == i);
      assert(hash_chaining_search(chaining, node->key));
      node = node->next;
    }
  }
}

int main(void) {
  hash_chaining *chaining = hash_chaining_init(MAXN);

  for (int i = 0; i < MAXN; ++i) {
    int val = rand() % 1000000;
    hash_chaining_insert(chaining, val);
    inserted[i] = val;
  }
  for (uint32_t i = 0; i < chaining->size; ++i) {
    list_node *node = chaining->slots[i];
    while (node) {
      in_chaining[in_chaining_cnt++] = node->key;
      node = node->next;
    }
  }

  assert(in_chaining_cnt == inserted_cnt);

  check_set_equal(inserted, inserted_cnt, in_chaining, in_chaining_cnt);
  check_chaining_property(chaining);

  puts("Hash with chaining tests passed.");

  fks_level1 *fks = fks_level1_build(chaining);

  for (uint32_t i = 0; i < fks->size; ++i) {
    fks_level2 *l2 = fks->level2_tables[i];
    list_node *node = chaining->slots[i];
    assert((l2 == NULL) == (node == NULL));
    if (l2 == NULL)
      continue;

    in_fks_cnt = 0;
    in_chaining_cnt = 0;
    for (uint32_t j = 0; j < l2->size; ++j) {
      if (l2->slots[j] != FKS_LEVEL2_EMPTY) {
        in_fks[in_fks_cnt++] = l2->slots[j];
        assert(hash_func(l2->slots[j], l2->parameters, l2->size) == j);
        assert(fks_level1_search(fks, l2->slots[j]));
      }
    }
    while (node) {
      in_chaining[in_chaining_cnt++] = node->key;
      node = node->next;
    }

    check_set_equal(in_fks, in_fks_cnt, in_chaining, in_chaining_cnt);
  }

  puts("FKS hash tests passed.");

  hash_chaining_destroy(chaining);
  fks_level1_destroy(fks);
  return 0;
}
