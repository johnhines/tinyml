
/*
 * SPDX-FileCopyrightText: 2026 John Hines
 * SPDX-License-Identifier: MIT
 *
 * @file neuron.c
 *
 * @brief Implements a simple neuron for learning purposes.
 *
 * @details Implemented in C with a minimum of library calls.
 *
 * @author John Hines
 * @date 2026-09-18
 *
 */

#include <stdio.h>
#include <stddef.h>

int main(void) {

    
  double x0_data[] = {0.0, 1.0, 0.0, 1.0, 2.0, 1.0};
  double x1_data[] = {0.0, 0.0, 1.0, 1.0, 1.0, 2.0};
  double y_data[]  = {1.0, 3.0, -2.0, 0.0, 2.0, -3.0};

  size_t n = 6;
  size_t epochs = 1000;

  double x0 = 0.0;
  double x1 = 0.0;
  double y = 0.0;

  double w0 = 0.5;
  double w1 = -0.5;
  double b = 0.1;

  double learning_rate = 0.01;
  double total_loss = 0.0;
  double mean_loss = 0.0;


  for( size_t epoch = 0; epoch < epochs; ++epoch) {
    total_loss = 0.0;

    for (size_t i = 0; i < n; ++i) {

        /* Initialize Training Data */ 
        x0 = x0_data[i];
        x1 = x1_data[i];
        y = y_data[i];

        /* forward pass */
        double y_hat = (w0 * x0) + (w1 * x1) + b;
        double error = y_hat - y;
        double loss = error * error;
      
        /* Backpropagation Pass */
          double dl_dy_hat = 2.0 * error;
          double dl_dw0 = dl_dy_hat * x0;
          double dl_dw1 = dl_dy_hat * x1;
          double dl_db = dl_dy_hat;

        /* Update Parameters */ 
        w0 -= learning_rate * dl_dw0;
        w1 -= learning_rate * dl_dw1;
        b -= learning_rate * dl_db;

        total_loss += loss;
        mean_loss = total_loss / (double)n;

      }
      if (epoch % 100 == 0) {
        printf("epoch %zu: loss = %.10f\n", epoch, mean_loss);
      }
  }

  printf("\nlearned parameters:\n");
  printf("w0 = %.6f\n", w0);
  printf("w1 = %.6f\n", w1);
  printf("b  = %.6f\n", b);
  printf("\n");

  return (0);
}
