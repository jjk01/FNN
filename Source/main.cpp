#include <iostream>
#include <ctime>
#include <stdlib.h>
#include "training_algorithm.h"
#include "data.h"

using namespace Eigen;


double accuracy(const MatrixXf & prediction, const RowVectorXi & classIndex) {
    MatrixXf::Index   maxIndex;
    double correct = 0;
    double total = classIndex.cols();

    for(int i = 0; i < total; ++i){
        prediction.col(i).maxCoeff( &maxIndex);
        if (maxIndex==classIndex(i)) ++correct;
    }
    return 100.*correct/total;
}


double accuracy(const MatrixXf & prediction, const MatrixXf    & target    ) {
    MatrixXf::Index   maxIndex_1, maxIndex_2;
    double correct = 0;
    double total = target.cols();

    for(int i = 0; i < total; ++i){
        prediction.col(i).maxCoeff( &maxIndex_1);
        target.col(i).maxCoeff( &maxIndex_2);
        if (maxIndex_1==maxIndex_2) ++correct;
    }

    return 100.*correct/total;
}


void loadNormalisedData(std::string path, MatrixXf& X){
    Matrix<unsigned char,-1,-1> X_load;
    read_images(path,X_load);

    X = X_load.template cast<float>();

    float mu = X.mean();
    X = X.array() - mu;
    float std = X.norm()/std::sqrt(X.size()-1);
    X /= std;
}


int main(){

    std::string train_img_path   = "/home/jamesklatzow/Documents/Machine_Learning/MNIST_data/train-images.idx3-ubyte";
    std::string train_label_path = "/home/jamesklatzow/Documents/Machine_Learning/MNIST_data/train-labels.idx1-ubyte";
    std::string test_img_path    = "/home/jamesklatzow/Documents/Machine_Learning/MNIST_data/t10k-images.idx3-ubyte";
    std::string test_label_path  = "/home/jamesklatzow/Documents/Machine_Learning/MNIST_data/t10k-labels.idx1-ubyte";


    neural_net N(784);
    N.addDense(32);
    N.addActivation(ActivationType::sigmoid);
    N.addDense(10);
    N.addActivation(ActivationType::sigmoid);

    MatrixXf Xtrain, Xtest;
    loadNormalisedData(train_img_path,Xtrain);

    std::vector<int> temp;
    read_label(train_label_path,temp);
    RowVectorXi Ytrain = RowVectorXi::Zero(int(temp.size()));

    for (unsigned n = 0; n < temp.size(); ++n){
        Ytrain(n) = temp[n];
    }


    Trainer<MatrixXf, RowVectorXi> T(&N,LossType::quadratic);
    double start =  std::clock();
    try {
       std::cout << "starting training\n\n";
       T.train(Xtrain,Ytrain);
    } catch (Exception & e) {
        e.what();
    }

    loadNormalisedData(test_img_path,Xtest);
    temp.clear();
    read_label(test_label_path,temp);
    RowVectorXi Ytest = RowVectorXi::Zero(int(temp.size()));

    for (unsigned n = 0; n < temp.size(); ++n){
        Ytest(n) = temp[n];
    }

    MatrixXf Z = N.propagate(Xtest);

    std::cout << accuracy(Z,Ytest) << "\n\n";

    std::cout << (std::clock() - start)/(double(CLOCKS_PER_SEC)) << "\n";

    return 0;
}
