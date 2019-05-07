#include "neural_layer.h"
#include "activation_functions.h"
#include "exception.h"
#include <cmath>
#include <random>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////


Layer::Layer(Layer * _previous): previous(_previous){}



ActivationLayer::ActivationLayer(ActivationType type, Layer * _previous): Layer(_previous), fn_type(type) {
    switch (type){
        case ActivationType::sigmoid:
            this->fn = &sigmoid_function;
            break;
        case ActivationType::ReLU:
            this->fn = &ReLU_function;
            break;
        case ActivationType::softmax:
            this->fn = &softmax_function;
            break;
        case ActivationType::identity:
            this->fn = &identity_function;
            break;
    }
}

std::unique_ptr<Layer> ActivationLayer::clone(Layer * _previous) {
    std::unique_ptr<Layer> l = std::make_unique<ActivationLayer>(fn_type,_previous);
    return std::move(l);
}



ActivationType ActivationLayer::funcType() {
    return fn_type;
}


LayerType ActivationLayer::type(void) { return LayerType::activation;}


MatrixXf ActivationLayer::feedForward(const MatrixXf & x) {
    return fn(x);
}


long ActivationLayer::inputSize() const {
    if (previous == nullptr) return 0;
    return previous -> outputSize();
}


long ActivationLayer::outputSize() const {
    return inputSize();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////


DenseLayer::DenseLayer(long n_in, long n_out, Layer * _previous): Layer(_previous),
    w(MatrixXf::Zero(n_out,n_in)), b(VectorXf::Zero(n_out)){

    std::default_random_engine generator;
    std::normal_distribution<double> weight_dist(0,1/sqrt(n_in)),bias_dist(0,1) ;

    for (long r = 0; r < n_out; ++r){
        b(r) = bias_dist(generator);
        for (long c = 0; c < n_in; ++c){
            w(r,c) = weight_dist(generator);
        }
    }
}

DenseLayer::DenseLayer(const DenseLayer & arg, Layer * prev): Layer(prev) {
    w = arg.w;
    b = arg.b;
}


std::unique_ptr<Layer> DenseLayer::clone(Layer * _previous) {
    std::unique_ptr<Layer> l = std::make_unique<DenseLayer>(*this, _previous);
    return std::move(l);
}



void DenseLayer::update(const MatrixXf& dw, const VectorXf & db){
    w += dw;
    b += db;
}

const MatrixXf & DenseLayer::weight() const {
    return w;
}


const VectorXf & DenseLayer::bias() const {
    return b;
}


LayerType DenseLayer::type(void) { return LayerType::dense;}


MatrixXf DenseLayer::feedForward(const MatrixXf & x) {
    return (w*x).colwise() + b;
}



long DenseLayer::inputSize() const{
    return w.cols();
}


long DenseLayer::outputSize() const{
    return w.rows();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
