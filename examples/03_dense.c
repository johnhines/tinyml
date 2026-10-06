

#include <stddef.h>
#include <stdio.h>

#include "../include/matrix.h"

int main(void) {
  /* define matrices & vectors for z = Wx + b */
  Matrix W = matrix_create(2, 3);
  Matrix x = matrix_create(3, 1);
  Matrix b = matrix_create(2, 1);
  Matrix z = matrix_create(2, 1);
  Matrix Wx = matrix_create(2, 1);

  /* initialize W */
  matrix_set(&W, 0, 0, 1);
  matrix_set(&W, 0, 1, 2);
  matrix_set(&W, 0, 2, 3);
  matrix_set(&W, 1, 0, 4);
  matrix_set(&W, 1, 1, 5);
  matrix_set(&W, 1, 2, 6);

  /* initialize x */
  matrix_set(&x, 0, 0, 2);
  matrix_set(&x, 1, 0, -1);
  matrix_set(&x, 2, 0, 3);

  /* initialize b */
  matrix_set(&b, 0, 0, 0.5);
  matrix_set(&b, 1, 0, -0.5);

  /* print as a check init is working */
  printf("\nWeights Matrix W\n");
  matrix_print(&W);
  printf("Input Vector x\n");
  matrix_print(&x);
  printf("Bias Vector b\n");
  matrix_print(&b);

  matrix_mul(&W, &x, &Wx);  // calculate Wx
  matrix_add(&Wx, &b, &z);  // calculate z = Wx +b

  printf("z = Wx + b\n");
  matrix_print(&z);  // print z as a check

  return (0);
}
