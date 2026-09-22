
/*
 * SPDX-FileCopyrightText: 2026 John Hines
 * SPDX-License-Identifier: MIT
 *
 * @file filename.cpp
 *
 * @brief Brief one-sentence description of what this file does.
 *
 * @details Detailed explanation of the module's implementation, algorithms
 * used, hardware interactions, or design patterns if necessary.
 *
 * @author John Hines
 * @date 2026-09-17
 *
 */

#include <stddef.h>
#include <stdio.h>

/* vector dot product */
static double dot(const double* a, const double* b, size_t n) {
  double result = 0;

  for (size_t i = 0; i < n; ++i) {
    result += a[i] * b[i];
  }

  return (result);
}

/* Matrix Vector Multiplication */
static void matvec(const double* matrix, const double* vector, double* output,
                   size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    output[i] = dot(&matrix[cols * i], &vector[0], cols);
  }
}

/* maxtrix Multiplication */
static void matmul(const double* A, const double* B, double* C, size_t m,
                   size_t n, size_t p) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      C[i * p + j] = 0;
      for (size_t k = 0; k < n; ++k) {
        C[i * p + j] += A[i * n + k] * B[k * p + j];
      }
    }
  }
}

/* Main() but actually just used for testing; this func will be deleted */
int main(void) {
  /* dot product test, no longer needed
  double x[] = {2.0, 3.0};
  double w[] = {0.5, -0.5};
  double bias = 0.1;

  double z = dot(x, w, 2) + bias;
  printf("z = %.6f\n", z);
  */

  double matrix[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double B[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  double C[4];

  /* matrix Multiplication test */

  matmul(A, B, C, 2, 3, 2);
  printf("\nmatmul C: %.1f %.1f %.1f %.1f\n", C[0], C[1], C[2], C[3]);

  double vector[] = {10.0, 20.0};

  double output[3];

  matvec(matrix, vector, output, 3, 2);
  printf("\nmatvec output[] = %.6f, %.6f, %.6f\n", output[0], output[1],
         output[2]);

  return (0);
}
