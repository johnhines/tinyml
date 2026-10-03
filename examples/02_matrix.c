

#include <stddef.h>
#include <stdio.h>

#include "matrix.h"

int main(void) {
  Matrix a = matrix_create(2, 3);
  Matrix b = matrix_create(3, 2);
  Matrix c = matrix_create(2, 2);

  /* initialize a */

  matrix_set(&a, 0, 0, 1);
  matrix_set(&a, 0, 1, 2);
  matrix_set(&a, 0, 2, 3);
  matrix_set(&a, 1, 0, 4);
  matrix_set(&a, 1, 1, 5);
  matrix_set(&a, 1, 2, 6);

  printf("Matrix A:");
  matrix_print(&a);

  /* initialize b */

  matrix_set(&b, 0, 0, 7);
  matrix_set(&b, 0, 1, 8);
  matrix_set(&b, 1, 0, 9);
  matrix_set(&b, 1, 1, 10);
  matrix_set(&b, 2, 0, 11);
  matrix_set(&b, 2, 1, 12);

  printf("Matrix B:");
  matrix_print(&b);

  matrix_mul(&a, &b, &c);

  matrix_print(&c);

  matrix_free(&a);
  matrix_free(&b);
  matrix_free(&c);

  return (0);
}
