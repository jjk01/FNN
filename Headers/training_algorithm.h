#ifndef TRAINING_ALGORITHM_H
#define TRAINING_ALGORITHM_H

#include "loss.h"
#include "gradient.h"
#include "neural_net.h"
#include <iostream>
#include <functional>

class Trainer;

class processBatch {
public:
    processBatch() = default;
    processBatch(neural_net * , Loss *, float rate, bool normalise = false);

    template <class Tin, class Tout>
    void iterate(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y, unsigned start, unsigned batch_size);

    float getRate(void);
    void setRate(float rate);
    void normalise(MatrixXf & xdata);
    
private:

    MatrixXf forwardPropagate(const MatrixXf & xdata);
    void backPropagate(const MatrixXf & yerror);

    std::vector<std::unique_ptr<Gradient>> layers;
    Loss * m_loss = nullptr;
    MatrixXf X,Z,error;
    float m_rate;
    bool m_normalise = false;
};




class Trainer {
public:

    Trainer(neural_net * , LossType, unsigned epochs = 25, unsigned batch_size = 32, float rate = 0.1f, bool batch_normalise = false);

    template <class Tin, class Tout>
    void train(const MatrixBase<Tin> & xdata, const MatrixBase<Tout> & ydata);

    template <class Tin, class Tout>
    void train(const MatrixBase<Tin> & xtrain, const MatrixBase<Tout> & ytrain,
               const MatrixBase<Tin> & xval,   const MatrixBase<Tout> & yval,
               float(*accuracy)(const MatrixXf&, const Tout&) = nullptr);

private:

    void printProgress(float loss, long step);
    void printProgress(float loss, float accu, long step);

    template <class Tin, class Tout>
    void shuffle(MatrixBase<Tin> & xdata, MatrixBase<Tout> & ydata);

    template <class Tin, class Tout>
    void processEpoch(const MatrixBase<Tin> & xdata, const MatrixBase<Tout> & ydata);

    template <class Tin, class Tout>
    float loss(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y);

    PermutationMatrix<Dynamic,Dynamic> P;

    processBatch m_batch;
    unsigned m_epochs{100};
    unsigned m_batch_size{32};

    neural_net * m_net = nullptr;
    std::unique_ptr<Loss> m_loss = nullptr;
    net_parameters optimal_parameters;

};




/*=================================================================================================================*/
/*=================================================================================================================*/

/* Template Implementations */



template <class Tin, class Tout>
void Trainer::train(const MatrixBase<Tin>& _xdata, const MatrixBase<Tout>& _ydata) {
    Tin  xdata(_xdata);
    Tout ydata(_ydata);
    float epoch_loss;

    for (long n = 0; n < m_epochs; ++n){
        shuffle(xdata,ydata);
        processEpoch(xdata,ydata);
        epoch_loss = loss(xdata,ydata);
        printProgress(epoch_loss,n);
    }
    std::cout << "Progress |" << std::string(50, '=') + ">| 100 %" << std::endl;
}




template <class Tin, class Tout>
void Trainer::train(const MatrixBase<Tin> & xtrain, const MatrixBase<Tout> & ytrain,
                    const MatrixBase<Tin> & xval,   const MatrixBase<Tout> & yval,
                    float(*func)(const MatrixXf&, const Tout&)) {

    Tin  xdata(xtrain);
    Tout ydata(ytrain);

    float accuracy, epoch_loss, max_accuracy = 0;
    MatrixXf xvalidation = xval.template cast<float>();
    m_batch.normalise(xvalidation);
    MatrixXf prediction;

    for (long n = 0; n < m_epochs; ++n){
        shuffle(xdata,ydata);
        processEpoch(xdata,ydata);
        epoch_loss = loss(xdata,ydata);

        if (func != nullptr) {
            prediction = m_net -> propagate(xvalidation);
            accuracy = func(prediction,yval);
            printProgress(epoch_loss,accuracy,n);
        } else {
            accuracy = 1 - loss(xval,yval);
            printProgress(epoch_loss,n);
        }

        if (max_accuracy < accuracy) {
            optimal_parameters = m_net -> Parameters();
            max_accuracy = accuracy;
        }
    }

    m_net -> setParameters(optimal_parameters);
    std::cout << "Progress |" << std::string(50, '=') + ">| 100 %" << std::endl;

}



template <class Tin, class Tout>
void Trainer::shuffle(MatrixBase<Tin> & xdata, MatrixBase<Tout> & ydata){
    P.resize(xdata.cols());
    P.setIdentity();
    std::random_shuffle(P.indices().data(), P.indices().data() + P.indices().size());
    xdata *= P;
    ydata *= P;
}



template <class Tin, class Tout>
void Trainer::processEpoch(const MatrixBase<Tin>& xdata, const MatrixBase<Tout>& ydata){
    unsigned batches = xdata.cols()/m_batch_size;

    for (unsigned n = 0; n < batches; ++n) {
        m_batch.iterate(xdata,ydata,n,m_batch_size);
    }

    unsigned remaining = xdata.cols()%m_batch_size;

    if (remaining != 0) {
        m_batch.iterate(xdata,ydata, m_batch_size*batches, remaining);
    }
}



template <class Tin, class Tout>
float Trainer::loss(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y){
    MatrixXf X = x.template cast<float> ();
    Tout Y(y);
    MatrixXf prediction = m_net -> propagate(X);;
    return m_loss -> loss(prediction,Y);
}



template <class Tin, class Tout>
void processBatch::iterate(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y, unsigned start, unsigned batch_size) {
    X = x.block(0,start,x.rows(),batch_size). template cast<float> ();
    if (m_normalise) normalise(X);
    Tout Y = y.block(0,start,y.rows(),batch_size);
    Z = forwardPropagate(X);
    error = m_loss -> error(Z,Y);
    backPropagate(error);
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif /* TRAINING_ALGORITHM_H */
