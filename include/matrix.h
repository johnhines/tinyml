
#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
  size_t rows;
  size_t cols;
  double* data;
} Matrix;

#endif

Matrix matrix_create(size_t rows, size_t cols);

void matrix_free(Matrix* matrix);

double matrix_get(const Matrix* matrix, size_t row, size_t col);

void matrix_set(Matrix* matrix, size_t row, size_t col, double value);

void matrix_print(const Matrix* matrix);

void matrix_mul(const Matrix* a, const Matrix* b, Matrix* out);

void matrix_add(const Matrix* a, const Matrix* b, Matrix* out);

void matrix_transpose(const Matrix* a, Matrix* out);
