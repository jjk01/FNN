#ifndef TRAINING_ALGORITHM_H
#define TRAINING_ALGORITHM_H

#include "loss.h"
#include "gradient.h"
#include "neural_net.h"
#include <iostream>



class iterateBatch {
public:
    iterateBatch(neural_net * , LossType, double rate);

    template <class Tin, class Tout>
    float operator()(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y, unsigned start, unsigned batch_size);

protected:
    MatrixXf forwardPropagate(const MatrixXf & xdata);
    void backPropagate(const MatrixXf & yerror);

    void batchNormalise(MatrixXf & xdata);

    std::vector<std::unique_ptr<Gradient>> layers;
    std::unique_ptr<Loss> loss = nullptr;

    MatrixXf X,Z,error;
    double m_rate;
};







class Trainer {
public:

    Trainer(neural_net * , LossType, unsigned epochs = 25, unsigned batch_size = 32, double rate = 0.1);

    template <class Tin, class Tout>
    void train(const MatrixBase<Tin> & xdata, const MatrixBase<Tout> & ydata);

    template <class Tin, class Tout>
    void train(const MatrixBase<Tin> & xtrain, const MatrixBase<Tout> & ytrain,
               const MatrixBase<Tin> & xval,   const MatrixBase<Tout> & yval);

private:

    void printProgress(float loss, long step);

    template <class Tin, class Tout>
    void shuffle(MatrixBase<Tin> & xdata, MatrixBase<Tout> & ydata);

    template <class Tin, class Tout>
    float processEpoch(const MatrixBase<Tin> & xdata, const MatrixBase<Tout> & ydata);

    PermutationMatrix<Dynamic,Dynamic> P;

    iterateBatch iterate;
    unsigned m_epochs{100};
    unsigned m_batch_size{32};
};




/*=================================================================================================================*/
/*=================================================================================================================*/

/* Template Implementations */




template <class Tin, class Tout>
void Trainer::train(const MatrixBase<Tin>& _xdata, const MatrixBase<Tout>& _ydata) {
    Tin  xdata(_xdata);
    Tout ydata(_ydata);

    float loss;


    for (long n = 0; n < m_epochs; ++n){
        shuffle(xdata,ydata);
        loss = processEpoch(xdata,ydata);
        printProgress(loss,n);
    }

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
float Trainer::processEpoch(const MatrixBase<Tin>& xdata, const MatrixBase<Tout>& ydata){

    unsigned batches = xdata.cols()/m_batch_size;
    float loss = 0;
    for (unsigned n = 0; n < batches; ++n) loss += iterate(xdata,ydata,n,m_batch_size);

    unsigned remaining = xdata.cols()%m_batch_size;
    if (remaining != 0) loss += iterate(xdata,ydata, m_batch_size*batches, remaining);

    return loss/float(batches);
}




template <class Tin, class Tout>
float iterateBatch::operator()(const MatrixBase<Tin> & x, const MatrixBase<Tout> & y, unsigned start, unsigned batch_size) {

    X = x.block(0,start,x.rows(),batch_size). template cast<float> ();
    Tout Y = y.block(0,start,y.rows(),batch_size);

    Z = forwardPropagate(X);
    error = loss -> error(Z,Y);
    backPropagate(error);

    return loss -> loss(Z,Y);
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#endif /* TRAINING_ALGORITHM_H */
