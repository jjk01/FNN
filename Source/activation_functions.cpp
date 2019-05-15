#include "activation_functions.h"


void identity_function(const MatrixXf& x, MatrixXf& y){
    y = x;
}



void sigmoid_function(const MatrixXf& x, MatrixXf& y){
    y = (1 + (-x.array()).exp()).inverse().matrix();
}


void tanh_function(const MatrixXf& x, MatrixXf& y){
    y = x.array().tanh();
}


void ReLU_function (const MatrixXf& x, MatrixXf& y){
    y = x.cwiseMax(0);
}



void softmax_function (const MatrixXf& x, MatrixXf& y){
    y = (x.array() - x.maxCoeff()).exp().matrix();
    VectorXf norm = y.colwise().sum().cwiseInverse();
    y *= norm.asDiagonal();
}


void identity_function(MatrixXf& x){
    // do nothing
}



void sigmoid_function(MatrixXf& x){
    x = (1 + (-x.array()).exp()).inverse().matrix();
}


void tanh_function(MatrixXf& x){
    x = x.array().tanh();
}


void ReLU_function (MatrixXf& x){
    x = x.cwiseMax(0);
}



void softmax_function (MatrixXf& x){
    x = (x.array() - x.maxCoeff()).exp().matrix();
    VectorXf norm = x.colwise().sum().cwiseInverse();
    x *= norm.asDiagonal();
}
