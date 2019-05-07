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

    /*
    # outer product of the grads and inputs
      grads_wrt_weights = np.matmul(grads_wrt_outputs[:, :, np.newaxis], inputs[:, np.newaxis, :])
      grads_wrt_weights = np.mean(grads_wrt_weights, axis = 0)
      grads_wrt_biases  = np.mean(grads_wrt_outputs, axis = 0)
      return [grads_wrt_weights,grads_wrt_biases]
    */
    long N = error.cols();
    MatrixXf dw = (r/N)*error*(input.transpose());
    MatrixXf db = r*error.rowwise().mean();
    layer -> update(dw,db);
}



MatrixXf identity_gradient(const MatrixXf& g, const MatrixXf& y){
    return g;
}



MatrixXf sigmoid_gradient (const MatrixXf& g, const MatrixXf& y){
    return ((1 - y.array())*(y.array())*(g.array())).matrix();
}



MatrixXf ReLU_gradient (const MatrixXf& g, const MatrixXf& y){
    return ((g.array())*(y.unaryExpr([](float x) -> float { return (x < 0) ? 0:1;}).array())).matrix();
}



MatrixXf softmax_gradient(const MatrixXf& Gy, const MatrixXf& y) {


    /*
    t2 = np.matmul(outputs[:, :, np.newaxis], outputs[:, np.newaxis, :])
        t1 = np.zeros(t2.shape)

        for ind in range(t1.shape[0]):
            np.fill_diagonal(t1[ind,:,:],outputs[ind,:])

        y = np.matmul(t1-t2,grads_wrt_outputs[:,:,np.newaxis])
        return np.squeeze(y)
    */

    MatrixXf D,U, Gx(Gy.rows(),Gy.cols());

    for (long n = 0; n < y.cols(); ++n){
        const MatrixXf & u = y.col(n);
        D = u.asDiagonal();
        U = -u*u.transpose();
        Gx.col(n) = (D+U)*Gy(n);
    }
    return Gx;
}

LayerType DenseGradient::type() {
    return LayerType::dense;
}
