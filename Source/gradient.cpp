#include "gradient.h"
#include <iostream>



ActivationGradient::ActivationGradient(ActivationLayer * _layer): layer(_layer) {

    ActivationType type = layer -> funcType();

    switch (type){
        case ActivationType::sigmoid:
            this->fn = &sigmoid_gradient;
            break;
        case ActivationType::ReLU:
            this->fn = &ReLU_gradient;
            break;
        case ActivationType::softmax:
            this->fn = &softmax_gradient;
            break;
        case ActivationType::identity:
            this->fn = &identity_gradient;
            break;
    }
}

LayerType ActivationGradient::type() {
    return LayerType::activation;
}

const MatrixXf & ActivationGradient::feedForward(const MatrixXf & x) {
    input = x;
    output = layer -> feedForward(x);
    return output;
}


MatrixXf ActivationGradient::feedBack(const MatrixXf & g) {
    error = g;
    return fn(g,output);
}


void ActivationGradient::update(float r){
    // Norhing need be done.
}


DenseGradient::DenseGradient(DenseLayer * _layer): layer(_layer){}


const MatrixXf & DenseGradient::feedForward(const MatrixXf & x) {
    input = x;
    output = layer -> feedForward(x);
    return output;
}


MatrixXf DenseGradient::feedBack(const MatrixXf & g) {
    error = g;
    const MatrixXf & W = layer -> weight();
    return W.transpose()*g;
}



void DenseGradient::update(float r) {
    long N = error.cols();
    MatrixXf dw = -(r/N)*error*(input.transpose());
    MatrixXf db = -r*error.rowwise().mean();
    layer -> update(dw,db);
}



MatrixXf identity_gradient(const MatrixXf& g, const MatrixXf& y){
    return g;
}



MatrixXf sigmoid_gradient (const MatrixXf& g, const MatrixXf& y){
    return (1 - y.array())*y.array()*g.array();
}



MatrixXf ReLU_gradient (const MatrixXf& g, const MatrixXf& y){
    MatrixXf d = y.unaryExpr([](float x) -> float { return (x < 0) ? 0:1;});
    return g.cwiseProduct(d);
}



MatrixXf softmax_gradient(const MatrixXf& Gy, const MatrixXf& y) {

    MatrixXf D,U, Gx(Gy.rows(),Gy.cols());

    for (long n = 0; n < y.cols(); ++n){
        const MatrixXf & u = y.col(n);
        const MatrixXf & g = Gy.col(n);
        D = u.asDiagonal();
        U = -u*u.transpose();
        Gx.col(n) = (D+U)*g;
    }
    return Gx;
}

LayerType DenseGradient::type() {
    return LayerType::dense;
}
