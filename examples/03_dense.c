

#include <stddef.h>
#include <stdio.h>

#include "../include/matrix.h"

int main(void) {
  Matrix W = matrix_create(2, 3);
  Matrix x = matrix_create(3, 1);
  Matrix b = matrix_create(2, 1);
  Matrix z = matrix_create(2, 1);
  Matrix Wx = matrix_create(2, 1);

  matrix_set(&W, 0, 0, 1);
  matrix_set(&W, 0, 1, 2);
  matrix_set(&W, 0, 2, 3);
  matrix_set(&W, 1, 0, 4);
  matrix_set(&W, 1, 1, 5);
  matrix_set(&W, 1, 2, 6);

  matrix_set(&x, 0, 0, 2);
  matrix_set(&x, 1, 0, -1);
  matrix_set(&x, 2, 0, 3);

  matrix_set(&b, 0, 0, 0.5);
  matrix_set(&b, 1, 0, -0.5);

  matrix_print(&W);
  matrix_print(&x);
  matrix_print(&b);

  matrix_mul(&W, &x, &Wx);
  matrix_add(&Wx, &b, &z);

  matrix_print(&z);

  return (0);
}
