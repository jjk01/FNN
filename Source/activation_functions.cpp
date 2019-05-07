#include "activation_functions.h"


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
    MatrixXf y = (x.array() - x.maxCoeff()).exp().matrix();
    VectorXf norm = y.colwise().sum().cwiseInverse();
    y *= norm.asDiagonal();
    return y;
}
