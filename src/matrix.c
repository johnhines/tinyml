#include "../include/matrix.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

Matrix matrix_create(size_t rows, size_t cols) {
  Matrix matrix;

  matrix.rows = rows;
  matrix.cols = cols;
  matrix.data = calloc(rows * cols, sizeof(double));

  if (matrix.data == NULL) {
    fprintf(stderr, "matrix_create(): memory allocatoin failed\n");
    exit(EXIT_FAILURE);
  }

  return matrix;
}

void matrix_free(Matrix* matrix) {
  // release the memory used for data
  free(matrix->data);

  // reset the vars just in case
  matrix->data = NULL;
  matrix->rows = 0;
  matrix->cols = 0;
}

double matrix_get(const Matrix* matrix, size_t row, size_t col) {
  return matrix->data[row * matrix->cols + col];
}

void matrix_set(Matrix* matrix, size_t row, size_t col, double value) {
  matrix->data[row * matrix->cols + col] = value;
}

void matrix_print(const Matrix* matrix) {
  for (size_t i = 0; i < matrix->rows; ++i) {
    for (size_t j = 0; j < matrix->cols; ++j) {
      printf("%8.2f ", matrix->data[i * matrix->cols + j]);
    }
  }
  printf("\n");
}

/* C (M x N) = A (M x K) * B (K x N), row-major. C must not overlap A or B. */
void matrix_mul(const Matrix* a, const Matrix* b, Matrix* out) {
  // check for invalid matrix shapes
  if (a->cols != b->rows || out->rows != a->rows || out->cols != b->cols) {
    fprintf(stderr, "matrix_mul(): incompatible dimensions\n");
    exit(EXIT_FAILURE);
  }

  // dereference struct elements into local vars
  size_t arows = a->rows;
  size_t bcols = b->cols;
  size_t acols = a->cols;
  double* adata = a->data;
  double* bdata = b->data;
  double* outdata = out->data;

  // fast multiply using pointers and indexes
  for (size_t i = 0; i < arows; i++) {
    for (size_t j = 0; j < bcols; j++) {
      double s = 0.0;
      for (size_t k = 0; k < acols; k++) {
        s += adata[i * acols + k] * bdata[k * bcols + j];
        outdata[i * bcols + j] = s;
      }
    }
  }
}

void matrix_add(const Matrix* a, const Matrix* b, Matrix* out) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "matrix_add(): incompatible dimensions\n");
    exit(EXIT_FAILURE);
  }

  size_t n = a->rows * a->cols;

  // assumes data is stored row-major order
  for (size_t i = 0; i < n; i++) {
    out->data[i] = a->data[i] + b->data[i];
  }
}

void matrix_transpose(const Matrix* a, Matrix* out) {
  size_t rows = a->rows;
  size_t cols = a->cols;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[j * rows + i] = a->data[i * cols + j];
    }
  }
}
