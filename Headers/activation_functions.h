#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;



MatrixXf identity_function(const MatrixXf & x);
MatrixXf sigmoid_function(const MatrixXf & x);
MatrixXf tanh_function (const MatrixXf & x);
MatrixXf ReLU_function (const MatrixXf & x);
MatrixXf softmax_function (const MatrixXf & x);



#endif /* ACTIVATION_FUNCTIONS_H */
