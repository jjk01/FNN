#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "neural_layer.h"
#include <memory>
#include <vector>

//class net_parameters;


class neural_net {
public:

    neural_net(long _input_size);
    neural_net(const neural_net &);
    neural_net & operator=(const neural_net &);

    void addActivation(ActivationType _type);
    void addDense(int Nout);

    MatrixXf propagate(const MatrixXf &);

    std::vector<Layer*> netLayers();


private:

    long m_input_size;
    std::vector<std::unique_ptr<Layer>> layers;
};






/*

z_{pqr}^{l} = sum_{x,y,z} W_{r,x,y,z}^{l} a_{x+p,y+q,z}^{l-1} + b_{pqr}^{l} = a.correlation(W) + b

epsilon_{i,j,k}^{l} = sigma'(z_{i,j,k}) sum_{x=0}^{min{i,n_x-1} sum_{y=0}^{min{j,n_y-1} sum_{z} W_{z,x,y,k}^{l+1} epsilon_{i-x,j-y,k}^{l+1}

epsilon_{i,j,k}^{l} = sigma'(z_{i,j,k}) epsilon^{l+1}.convolve(W_{k}^{l+1})

dC/d(b_{ijk}^{l}) = epsilon_{ijk}^{l}

dC/d(W_{r,x,y,z}^{l}) = sum_{i,j} epsilon_{ijr}^{l} a_{i+x,j+y,z}^{l-1}

*/


#endif /* NEURAL_NET_H */
