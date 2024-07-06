#include "../inc/hash_parameters.h"

hash_parameters generate_hash_parameters(void) {
  static uint32_t state_a = 5735423;
  static uint32_t state_b = 2725621;

  state_a = state_a * 6074147 + 8734093;
  state_b = state_b * 6463879 + 3346663;

  return (hash_parameters){.a = state_a, .b = state_b};
}
