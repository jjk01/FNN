#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "neural_layer.h"
#include <memory>
#include <vector>

struct net_parameters;


class neural_net {
public:

    neural_net(long _input_size);
    neural_net(const neural_net &);
    neural_net & operator=(const neural_net &);

    void addActivation(ActivationType _type);
    void addDense(int Nout);
    void setParameters(const net_parameters &);
    net_parameters Parameters(void) const;

    MatrixXf propagate(const MatrixXf &);
    std::vector<Layer*> netLayers();

private:
    long m_input_size;
    std::vector<std::unique_ptr<Layer>> layers;
};



struct net_parameters {
    std::vector<MatrixXf> w;
    std::vector<VectorXf> b;
};

#endif /* NEURAL_NET_H */
