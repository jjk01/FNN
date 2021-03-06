#include <iostream>
#include <ctime>
#include <stdlib.h>
#include "training_algorithm.h"
#include "data.h"

using namespace Eigen;




/*
 * To Do *
 *
 * 1) Clean up training interface. Add loss, variable rate etc.
 * 2) Batch normalisation.
 * 3) Commandline arguments.
 * 6) Implement weight saving.
 * 9) Addaptive rate.

 * Done
 *
 * 4) Find issues with ReLU and softmax.
 * 5) Include validation set and only use best weights.
 * 7) Add in tanh.
 * 8) Add in user defined accuracy.
 * */


float accuracy(const MatrixXf & prediction, const RowVectorXi & classIndex);
void loadNormalisedData(std::string img_path, std::string label_path, MatrixXf& X, RowVectorXi& Y);


int main(){

    std::string data_dir         =  "/home/jamesklatzow/Documents/Machine_Learning/MNIST_data/";

    std::string train_img_path   =  data_dir + "train-images.idx3-ubyte";
    std::string train_label_path =  data_dir + "train-labels.idx1-ubyte";
    std::string test_img_path    =  data_dir + "t10k-images.idx3-ubyte";
    std::string test_label_path  =  data_dir + "t10k-labels.idx1-ubyte";


    neural_net N(784);
    N.addDense(150);
    N.addActivation(ActivationType::tanh);
    N.addDense(10);


    MatrixXf Xtrain, Xtest;
    RowVectorXi Ytrain, Ytest;

    loadNormalisedData(train_img_path, train_label_path, Xtrain, Ytrain);
    loadNormalisedData(test_img_path, test_label_path,Xtest,Ytest);

    MatrixXf Z = N.propagate(Xtest);
    std::cout << "start accuracy = " << accuracy(Z,Ytest) << "\n\n";

    Trainer T(&N,LossType::cross_entropy_softmax,25,30,0.1f, true);

    double start =  std::clock();
    T.train(Xtrain,Ytrain, Xtest,Ytest,accuracy);
    double time = (std::clock() - start)/(double(CLOCKS_PER_SEC));

    N.addActivation(ActivationType::softmax);
    Z = N.propagate(Xtest);
    std::cout << "\n\n" << "accuracy = " << accuracy(Z,Ytest) << " in " << time << "s" << "\n\n";

    return 0;
}




float accuracy(const MatrixXf & prediction, const RowVectorXi & classIndex) {

    MatrixXf::Index   maxIndex;
    float correct = 0;
    float total = classIndex.cols();

    for(int i = 0; i < total; ++i){
        prediction.col(i).maxCoeff( &maxIndex);
        if (maxIndex==classIndex(i)) ++correct;
    }

    return 100.f*correct/total;
}



void loadNormalisedData(std::string img_path, std::string label_path, MatrixXf& X, RowVectorXi& Y){
    Matrix<unsigned char,-1,-1> X_load;
    read_images(img_path,X_load);

    X = X_load.template cast<float>();

    float mu = X.mean();
    X = X.array() - mu;
    float std = X.norm()/float(std::sqrt(X.size()-1));
    X /= std;

    std::vector<int> temp;
    read_label(label_path,temp);
    Y = RowVectorXi::Zero(int(temp.size()));

    for (unsigned n = 0; n < temp.size(); ++n){
        Y(n) = temp[n];
    }
}
