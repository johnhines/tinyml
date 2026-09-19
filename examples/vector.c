
/*
 * SPDX-FileCopyrightText: 2026 John Hines 
 * SPDX-License-Identifier: MIT
 *
 * @file filename.cpp
 *  
 * @brief Brief one-sentence description of what this file does.
 *
 * @details Detailed explanation of the module's implementation, algorithms used, 
 *          hardware interactions, or design patterns if necessary.
 *
 * @author John Hines
 * @date 2026-09-17
 * 
 */

#include  <stddef.h>
#include  <stdio.h>

static double dot(const double *a, const double *b, size_t n)
{
  double result = 0;

  for( size_t i = 0; i < n; ++i ) {
    result += a[i] * b[i];
  }

  return(result);
}


int main(void)
{
  double x[] = {2.0, 3.0};
  double w[] = {0.5, -0.5};
  double bias = 0.1;

  double z = dot(x, w, 2) + bias;

  printf("z = %.6f\n", z);

  return(0);
}

