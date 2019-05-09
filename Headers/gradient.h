#ifndef GRADIENT_HPP
#define GRADIENT_HPP


#include "neural_layer.h"


using namespace Eigen;


MatrixXf identity_gradient(const MatrixXf &, const MatrixXf&);
MatrixXf sigmoid_gradient(const MatrixXf &, const MatrixXf &);
MatrixXf tanh_gradient(const MatrixXf &, const MatrixXf &);
MatrixXf ReLU_gradient(const MatrixXf &, const MatrixXf &);
MatrixXf softmax_gradient(const MatrixXf &, const MatrixXf &);



class Gradient {
public:
    Gradient() = default;
    virtual ~Gradient() = default;
    virtual const MatrixXf & feedForward(const MatrixXf &) = 0;
    virtual MatrixXf feedBack(const MatrixXf &) = 0;
    virtual void update(float r) = 0;
    virtual LayerType type() = 0;
protected:
    MatrixXf input;
    MatrixXf output;
    MatrixXf error;
};



class ActivationGradient: public Gradient {
public:
    ActivationGradient(ActivationLayer *);
    const MatrixXf & feedForward(const MatrixXf &);
    MatrixXf feedBack(const MatrixXf &);
    void update(float r);
    LayerType type();
private:
    ActivationLayer * layer = nullptr;
    MatrixXf (*fn)(const MatrixXf& g, const MatrixXf& y);
};




class DenseGradient: public Gradient {
public:
    DenseGradient(DenseLayer *);
    const MatrixXf & feedForward(const MatrixXf &);
    MatrixXf feedBack(const MatrixXf &);
    void update(float r);
    LayerType type();
private:
    DenseLayer * layer = nullptr;
};



#endif /* GRADIENT_HPP */
