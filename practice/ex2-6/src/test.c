#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int N = 2000;    // length of the square region
int step = 2000; // Number of update steps
#define EPS 1e-5

void baseline(int N, int step, double *p, double *p_next);
void impl(int N, int step, double *p);

void display_time(struct timespec start, struct timespec end) {
  printf("%fms\n", (double)(end.tv_sec - start.tv_sec) * 1000.0f +
                       (double)(end.tv_nsec - start.tv_nsec) / 1000000.0f);
}

double is_legal_answer(int N, double *ref_p, double *ref_p_next, double *p) {
  double diff = 0;
  int flg = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(ref_p[i * N + j] - p[i * N + j]) <
          fabs(ref_p_next[i * N + j] - p[i * N + j])) {
        diff += (ref_p[i * N + j] - p[i * N + j]) *
                (ref_p[i * N + j] - p[i * N + j]);
      } else {
        diff += (ref_p_next[i * N + j] - p[i * N + j]) *
                (ref_p_next[i * N + j] - p[i * N + j]);
      }
      if (!flg && diff > EPS) {
        flg = 1;
        printf("diff > EPS since i: %d, j: %d\n", i, j);
      }
    }
  }
  return diff;
}

int main(void) {
  double *ref_p = calloc(N * N, sizeof(double));
  double *ref_p_next = calloc(N * N, sizeof(double));
  double *p = calloc(N * N, sizeof(double));

  // initial values for grid
  for (int i = 0; i < N; i++) {
    ref_p[i] = 1.0f;
  }
  for (int j = 0; j < N; j++) {
    ref_p[j * N] = 1.0f;
  }
  // for (int i = 0; i < N; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     ref_p[i * N + j] = rand() % 32;
  //   }
  // }

  memcpy(ref_p_next, ref_p, N * N * sizeof(double));
  memcpy(p, ref_p, N * N * sizeof(double));

  struct timespec start, end;

  // baseline
  clock_gettime(CLOCK_MONOTONIC, &start);
  baseline(N, step, ref_p, ref_p_next);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Baseline: ");
  display_time(start, end);

  // your implementation
  clock_gettime(CLOCK_MONOTONIC, &start);
  impl(N, step, p);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Yours:    ");
  display_time(start, end);
  double diff = is_legal_answer(N, ref_p, ref_p_next, p);
  if (diff <= EPS) {
    puts("-------Pass-------");
    printf("Diff: %f\n", diff);
  } else {
    puts("x-x-x-Invalid-x-x-x");
    printf("Diff: %f\n", diff);
  }

  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     printf("%f ", ref_p[i * N + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < N; j++) {
  //     printf("%f ", ref_p_next[i * N + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  free(ref_p);
  free(ref_p_next);
  free(p);
  return 0;
}
