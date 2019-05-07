#ifndef TRAINING_ALGORITHM_H
#define TRAINING_ALGORITHM_H

#include "loss.h"
#include "gradient.h"
#include "neural_net.h"
#include <iostream>


template <class Tin, class Tout>
class iterateBatch {
public:
    iterateBatch() = default;
    iterateBatch(neural_net * , LossType, double rate);
    float operator()(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y, unsigned start, unsigned batch_size);

private:
    MatrixXf forwardPropagate(const MatrixXf & xdata);
    void backPropagate(const MatrixXf & yerror);

    void batchNormalise(MatrixXf & xdata);

    std::vector<std::unique_ptr<Gradient>> layers;
    std::unique_ptr<Loss> loss = nullptr;

    MatrixXf X,Z,error;
    Tout Y;
    double m_rate;
};



template <class Tin, class Tout>
class Trainer {
public:
    Trainer() = default;
    Trainer(neural_net * , LossType, unsigned epochs = 25, unsigned batch_size = 32, double rate = 0.1);
    void train(const MatrixBase<Tin> & xdata, const MatrixBase<Tout> & ydata);

private:
    void shuffle(MatrixBase<Tin> & xdata, MatrixBase<Tout> & ydata);
    float processEpoch(const MatrixBase<Tin> & xdata, const MatrixBase<Tout> & ydata);

    PermutationMatrix<Dynamic,Dynamic> P;
    iterateBatch<Tin,Tout> iterate;

    unsigned m_epochs{100};
    unsigned m_batch_size{32};
};




/*=================================================================================================================*/
/*=================================================================================================================*/

/* Template Instantiations */

template <class Tin, class Tout>
Trainer<Tin,Tout>::Trainer(neural_net * net, LossType loss, unsigned epochs, unsigned batch_size, double rate):
    iterate(net, loss, rate), m_epochs(epochs), m_batch_size(batch_size){}



template <class Tin, class Tout>
void Trainer<Tin,Tout>::train(const MatrixBase<Tin>& _xdata, const MatrixBase<Tout>& _ydata) {
    Tin  xdata(_xdata);
    Tout ydata(_ydata);

    float loss;
    std::string to_write;

    for (long n = 0; n < m_epochs; ++n){

        shuffle(xdata,ydata);
        loss = processEpoch(xdata,ydata);

        long percentage = (100*n)/m_epochs;
        long barLength = percentage/2;
        std::string bar = std::string(barLength, '=') + ">" + std::string(50 - barLength, '_') ;
        std::ostringstream stringStream;
        stringStream << " Progress |" << bar + "| " << std::to_string(percentage) << " %, loss = " << loss;
        to_write = stringStream.str();
        std::cout << to_write << "\r" << std::flush;
    }
    std::cout << std::string(to_write.size(), ' ') << "\r";
    std::cout << "Progress |" << std::string(50, '=') + ">| 100 %\n\n";
}



template <class Tin, class Tout>
void Trainer<Tin,Tout>::shuffle(MatrixBase<Tin> & xdata, MatrixBase<Tout> & ydata){
    P.resize(xdata.cols());
    P.setIdentity();
    std::random_shuffle(P.indices().data(), P.indices().data() + P.indices().size());
    xdata *= P;
    ydata *= P;
}



template <class Tin, class Tout>
float Trainer<Tin,Tout>::processEpoch(const MatrixBase<Tin>& xdata, const MatrixBase<Tout>& ydata){

    unsigned batches = xdata.cols()/m_batch_size;
    float loss = 0;
    for (unsigned n = 0; n < batches; ++n) loss += iterate(xdata,ydata,n,m_batch_size);

    unsigned remaining = xdata.cols()%m_batch_size;
    if (remaining != 0) loss += iterate(xdata,ydata, m_batch_size*batches, remaining);

    return loss/float(batches);
}




template <class Tin, class Tout>
iterateBatch<Tin,Tout>::iterateBatch(neural_net * net, LossType _loss, double rate): m_rate(rate) {

    std::vector<Layer*> NN_layers = net -> netLayers();

    for (std::vector<Layer*>::iterator l = NN_layers.begin(); l != NN_layers.end(); ++l){

        LayerType type = (*l) -> type();

        switch (type){
            case LayerType::dense: {
                    DenseLayer * t = static_cast<DenseLayer*>(*l);
                    layers.push_back(std::make_unique<DenseGradient>(t));
                }
                break;
            case LayerType::activation: {
                    ActivationLayer * t = static_cast<ActivationLayer*>(*l);
                    layers.push_back(std::make_unique<ActivationGradient>(t));
                }
                break;
        }
    }

    switch (_loss) {
        case LossType::quadratic:
            loss.reset(new quadratic());
            break;
        case LossType::cross_entropy_softmax:
            loss.reset(new cross_entropy_softmax());
            break;
    }
}



template <class Tin, class Tout>
float iterateBatch<Tin,Tout>::operator()(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y, unsigned start, unsigned batch_size) {

    X = x.block(0,start,x.rows(),batch_size). template cast<float> ();
    Tout Y = y.block(0,start,y.rows(),batch_size);

    Z = forwardPropagate(X);
    error = loss -> error(Z,Y);
    backPropagate(error);

    return loss -> loss(Z,Y);
}


template <class Tin, class Tout>
void iterateBatch<Tin,Tout>::batchNormalise(MatrixXf & xdata) {
    float mu = xdata.mean();
    xdata = xdata.array() - mu;
    float std = xdata.norm()/std::sqrt(xdata.size()-1);
    xdata /= std;
}



template <class Tin, class Tout>
MatrixXf iterateBatch<Tin,Tout>::forwardPropagate(const MatrixXf & xdata){
    MatrixXf y(xdata);

    for (auto i = layers.begin(); i != layers.end(); ++i){
        y = (*i) -> feedForward(y);
    }
    return y;
}

template <class Tin, class Tout>
void iterateBatch<Tin,Tout>::backPropagate(const MatrixXf & yerror) {
    MatrixXf g(yerror);

    for (auto i = layers.rbegin(); i != layers.rend(); ++i){
        g = (*i) -> feedBack(g);
        (*i) -> update(m_rate);

    }
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif /* TRAINING_ALGORITHM_H */
