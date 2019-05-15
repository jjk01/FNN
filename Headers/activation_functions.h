#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;



void identity_function(const MatrixXf& x, MatrixXf& y);
void  sigmoid_function(const MatrixXf& x, MatrixXf& y);
void     tanh_function(const MatrixXf& x, MatrixXf& y);
void     ReLU_function(const MatrixXf& x, MatrixXf& y);
void  softmax_function(const MatrixXf& x, MatrixXf& y);


void identity_function(MatrixXf& );
void  sigmoid_function(MatrixXf& );
void     tanh_function(MatrixXf& );
void     ReLU_function(MatrixXf& );
void  softmax_function(MatrixXf& );


#endif /* ACTIVATION_FUNCTIONS_H */
