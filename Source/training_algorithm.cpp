#include "training_algorithm.h"




///////////////////////////////////////////////////////////////////////////////////////////////////////////////


Trainer::Trainer(neural_net * net, LossType loss, unsigned epochs, unsigned batch_size, float rate, bool batch_normalise):
    m_epochs(epochs), m_batch_size(batch_size), m_net(net){

    optimal_parameters =  m_net -> Parameters();

    switch (loss) {
        case LossType::quadratic:
            m_loss.reset(new quadratic());
            break;
        case LossType::cross_entropy_softmax:
            m_loss.reset(new cross_entropy_softmax());
            break;
    }

    m_batch = processBatch(net,m_loss.get(),rate, batch_normalise);

}



void Trainer::printProgress(float loss, long step){

    long percentage = (100*step)/m_epochs;
    unsigned long barLength = static_cast<unsigned long>(percentage/2);
    std::string bar = std::string(barLength, '=') + ">" + std::string(50 - barLength, '_') ;
    std::ostringstream stringStream;
    stringStream << " Progress |" << bar + "| " << std::to_string(percentage) << " %, loss = " << loss;
    std::cout << stringStream.str() << "\r" << std::flush;
}


void Trainer::printProgress(float loss, float acc, long step){

    long percentage = (100*step)/m_epochs;
    unsigned long barLength = static_cast<unsigned long>(percentage/2);
    std::string bar = std::string(barLength, '=') + ">" + std::string(50 - barLength, '_') ;
    std::ostringstream stringStream;
    stringStream << " Progress |" << bar + "| " << std::to_string(percentage) << " %, loss = " << loss << " , acc = " << acc;
    std::cout << stringStream.str() << "\r" << std::flush;
}




processBatch::processBatch(neural_net * net, Loss * loss, float rate, bool normalise): m_loss(loss),  m_rate(rate), m_normalise(normalise) {

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
}


void processBatch::setRate(float rate){
    m_rate = rate;
}


void processBatch::normalise(MatrixXf & xdata) {
    float mu = xdata.mean();
    xdata = xdata.array() - mu;
    float std = xdata.norm()/float(std::sqrt(xdata.size()-1));
    xdata /= std;
}




MatrixXf processBatch::forwardPropagate(const MatrixXf & xdata){
    MatrixXf y(xdata);
    for (auto i = layers.begin(); i != layers.end(); ++i){
        y = (*i) -> feedForward(y);
    }
    return y;
}


void processBatch::backPropagate(const MatrixXf & yerror) {
    MatrixXf g(yerror);

    for (auto i = layers.rbegin(); i != layers.rend(); ++i){
        g = (*i) -> feedBack(g);
        (*i) -> update(m_rate);

    }
}
