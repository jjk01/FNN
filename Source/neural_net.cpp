#include "neural_net.h"

#include <fstream>
#include <iostream>




neural_net::neural_net(long _input_size): m_input_size(_input_size){}


neural_net::neural_net(const neural_net & arg) {
    m_input_size = arg.m_input_size;
    layers.resize(arg.layers.size());
    Layer * previous = nullptr;

    for (unsigned long n = 0; n < layers.size(); ++n) {
        layers[n] = arg.layers[n] -> clone(previous);
        previous = layers[n].get();
    }
}



neural_net & neural_net::operator=(const neural_net & arg) {

    if (this != &arg) {
        m_input_size = arg.m_input_size;
        layers.clear();
        layers.resize(arg.layers.size());
        Layer * previous = nullptr;
        for (unsigned long n = 0; n < layers.size(); ++n) {
            layers[n] = arg.layers[n] -> clone(previous);
            previous = layers[n].get();
        }
    }
    return *this;
}



void neural_net::addActivation(ActivationType _type) {
    Layer * prev = nullptr;
    if (!layers.empty()) prev = layers.back().get();
    std::unique_ptr <Layer> l = std::make_unique<ActivationLayer>(_type, prev);
    layers.push_back(std::move(l));
}


void neural_net::addDense(int Nout) {
    long Nin = m_input_size;
    Layer * prev = nullptr;

    if (!layers.empty()){
        prev = layers.back().get();
        Nin = prev -> outputSize();
    }

    std::unique_ptr <Layer> l = std::make_unique<DenseLayer>(Nin,Nout,prev);
    layers.push_back(std::move(l));
}



MatrixXf neural_net::propagate(const MatrixXf & x){

    MatrixXf y(x);

    for (auto i = layers.begin(); i != layers.end(); ++i){
        y = (*i) -> feedForward(y);
    }

    return y;
}


std::vector<Layer*> neural_net::netLayers()  {
    std::vector<Layer*> _layers(layers.size(),nullptr);

    for (unsigned long l = 0; l < layers.size(); ++l){
        _layers[l] = layers[l].get();
    }
    return _layers;
}


void neural_net::setParameters(const net_parameters & parameters){

    unsigned long m = 0;
    for (unsigned long n = 0; n < layers.size(); ++n){

        LayerType type = layers[n] -> type();

        if (type == LayerType::dense){
            DenseLayer * t = static_cast<DenseLayer*>(layers[n].get());
            t -> setParameters(parameters.w[m], parameters.b[m]);
            ++m;
        }
    }
}


net_parameters neural_net::Parameters(void) const {
    net_parameters parameters;

    for (unsigned long n = 0; n < layers.size(); ++n){

        LayerType type = layers[n] -> type();

        if (type == LayerType::dense){
            DenseLayer * t = static_cast<DenseLayer*>(layers[n].get());
            parameters.w.push_back( t -> weight() );
            parameters.b.push_back( t -> bias() );
        }
    }

    return parameters;

}


/*===============================================================================================*/



