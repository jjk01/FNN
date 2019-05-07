#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;



MatrixXf identity_function(const MatrixXf & x){
    return x;
}



MatrixXf sigmoid_function(const MatrixXf & x){
    return (1 + (-x.array()).exp()).inverse().matrix();
}



MatrixXf ReLU_function (const MatrixXf & x){
    return x.cwiseMax(0);
}



MatrixXf softmax_function (const MatrixXf & x){
    MatrixXf y = (-x).array().exp().matrix();
    VectorXf norm = y.colwise().sum().cwiseInverse();
    y *= norm.asDiagonal();
    return y;
}






#endif /* ACTIVATION_FUNCTIONS_H */
