void baseline(int N, int step, double *p, double *p_next) {
  //   for (int k = 0; k < step; k++) {
  // #pragma omp parallel for
  //     for (int i = 1; i < N - 1; i++) {
  //       for (int j = 1; j < N - 1; j++) {
  //         double p1 = p[(i - 1) * N + j];
  //         double p2 = p[(i + 1) * N + j];
  //         double p3 = p[i * N + j + 1];
  //         double p4 = p[i * N + j - 1];
  //         p_next[i * N + j] = (p1 + p2 + p3 + p4) / 4.0f;
  //       }
  //     }
  //     double *temp = p;
  //     p = p_next;
  //     p_next = temp;
  //   }
}