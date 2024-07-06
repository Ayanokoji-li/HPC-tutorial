#include "../inc/hash_func.h"

uint32_t hash_func(uint32_t key, hash_parameters parameters, uint32_t m) {
  return ((parameters.a * key + parameters.b) % 100000007U) % m;
}
