#include "training_algorithm.h"




///////////////////////////////////////////////////////////////////////////////////////////////////////////////


Trainer::Trainer(neural_net * net, LossType loss, unsigned epochs, unsigned batch_size, double rate):
    iterate(net,loss,rate),
    m_epochs(epochs), m_batch_size(batch_size){}




void Trainer::printProgress(float loss, long step){

    long percentage = (100*step)/m_epochs;
    long barLength = percentage/2;
    std::string bar = std::string(barLength, '=') + ">" + std::string(50 - barLength, '_') ;
    std::ostringstream stringStream;
    stringStream << " Progress |" << bar + "| " << std::to_string(percentage) << " %, loss = " << loss;
    std::cout << stringStream.str() << "\r" << std::flush;
}




iterateBatch::iterateBatch(neural_net * net, LossType _loss, double rate): m_rate(rate) {

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


void iterateBatch::setRate(double rate){
    m_rate = rate;
}


void iterateBatch::batchNormalise(MatrixXf & xdata) {
    float mu = xdata.mean();
    xdata = xdata.array() - mu;
    float std = xdata.norm()/std::sqrt(xdata.size()-1);
    xdata /= std;
}




MatrixXf iterateBatch::forwardPropagate(const MatrixXf & xdata){
    MatrixXf y(xdata);

    for (auto i = layers.begin(); i != layers.end(); ++i){
        y = (*i) -> feedForward(y);
    }
    return y;
}


void iterateBatch::backPropagate(const MatrixXf & yerror) {
    MatrixXf g(yerror);

    for (auto i = layers.rbegin(); i != layers.rend(); ++i){
        g = (*i) -> feedBack(g);
        (*i) -> update(m_rate);

    }
}


