
/*
 * SPDX-FileCopyrightText: 2026 John Hines 
 * SPDX-License-Identifier: MIT
 *
 * @file neuron.c
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

#include <stdio.h>


static double compute_loss(double w0, double w1, double y, double x0, double x1, double b)
{
  double y_hat = w0 * x0 + w1 * x1 - b;
  double error = y_hat - y;

  return error * error;
}


int main(void)
{
  double x0 = 2.0;
  double x1 = 3.0;
  double y = -4.0;

  double w0 = 0.5;
  double w1 = -0.5;
  double b = 0.1;

  double learning_rate = 0.01;

  /*
  * Forward Pass 
  */

 double y_hat = w0*x0 + w1*x1 + b;

  /*
   * Squared Error Loss 
  */

 double error = y_hat - y;
 double loss = error*error;

 printf("\nPrediction = %f\n", y_hat);
 printf("Target     = %f\n", y);
 printf("Loss       = %f\n", loss);



  /* Backpropagation Pass
  */

  double dl_dy_hat = 2.0*error;
  
  double dl_dw0 = dl_dy_hat*x0;
  double dl_dw1 = dl_dy_hat*x1;
  double dl_db = dl_dy_hat;

  printf("\nGradients\n");
  printf("dl/dw0 = %f\n", dl_dw0);
  printf("dl/dw1 = %f\n", dl_dw1);
  printf("dl/db = %f\n", dl_db);

  w0 -= learning_rate*dl_dw0;
  w1 -= learning_rate*dl_dw1;
  b -= learning_rate*dl_db;

  printf("\nUpdated Parameters\n");
  printf("w0 = %f\n", w0);
  printf("w1 = %f\n", w1);
  printf("b  = %f\n", b);


  /* numerically check derivatives */

  double epsilon = 1e-5;
  
  double w0_loss_plus = compute_loss(x0,x1,y,w0+epsilon,w1,b);
  double w0_loss_minus = compute_loss(x0,x1,y,w0-epsilon,w1,b);
  double numerical_dw0 = (w0_loss_plus-w0_loss_minus)/(2*epsilon);
  
  double w1_loss_plus = compute_loss(x0,x1,y,w0,w1+epsilon,b);
  double w1_loss_minus = compute_loss(x0,x1,y,w0,w1-epsilon,b);
  double numerical_dw1 = (w1_loss_plus - w1_loss_minus) / (2 * epsilon);

  double b_loss_plus = compute_loss(x0,x1,y,w0,w1,b+epsilon);
  double b_loss_minus = compute_loss(x0,x1,y,w0,w1,b-epsilon);
  double numerical_db = (b_loss_plus - b_loss_minus) / (2 * epsilon);

  printf("\nGradient Check\n");
  printf("Parameter        Analytical        Numerical\n");
  printf("w0               %f         %f\n",dl_dw0,numerical_dw0);
  printf("w1               %f         %f\n",dl_dw1,numerical_dw1);
  printf("b                %f         %f\n\n",dl_db,numerical_db);

  
  return(0);
 
}
