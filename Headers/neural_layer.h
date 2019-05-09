#ifndef NEUTRON_LAYER_H
#define NEUTRON_LAYER_H

#include <Eigen/Dense>
#include <memory>

using namespace Eigen;



enum class ActivationType {
    identity,
    sigmoid,
    tanh,
    ReLU,
    softmax,
};


enum class LayerType {
    dense,
    activation,
};



class Layer {
public:

    Layer(Layer * _previous);
    virtual ~Layer() = default;

    virtual LayerType type(void) = 0;
    virtual MatrixXf feedForward(const MatrixXf &) = 0;
    virtual std::unique_ptr<Layer> clone(Layer * _previous) = 0;

    virtual long inputSize() const = 0;
    virtual long outputSize() const = 0;

protected:

    Layer * previous = nullptr;
};





class ActivationLayer: public Layer {
public:
    ActivationLayer(ActivationType _type, Layer * _previous = nullptr);
    std::unique_ptr<Layer> clone(Layer * _previous);

    MatrixXf feedForward(const MatrixXf &);

    LayerType type(void);
    ActivationType funcType();
    long inputSize() const;
    long outputSize() const;

protected:
    MatrixXf (*fn)(const MatrixXf&);
    ActivationType fn_type;
};




class DenseLayer: public Layer {
public:

    DenseLayer(long _Nin, long _Nout, Layer * _previous = nullptr);
    DenseLayer(const DenseLayer &, Layer *);

    std::unique_ptr<Layer> clone(Layer * _previous);

    LayerType type(void);
    MatrixXf feedForward(const MatrixXf &);

    void setParameters(const MatrixXf& w, const VectorXf & b);
    void update(const MatrixXf& dw, const VectorXf & db);
    const MatrixXf & weight(void) const;
    const VectorXf & bias(void) const;
    long inputSize() const;
    long outputSize() const;

protected:
    MatrixXf m_w;
    VectorXf m_b;
};





#endif /* NEUTRON_LAYER_H */
