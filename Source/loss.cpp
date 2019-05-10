#include "loss.h"
#include <iostream>
#include "activation_functions.h"



MatrixXf quadratic::error(const MatrixXf & prediction, const MatrixXf & target) {
    return (prediction - target);
}


float quadratic::loss(const MatrixXf & prediction, const MatrixXf & target) {
    float L = 0.5f*(prediction - target).colwise().squaredNorm().mean();
    return L;
}



MatrixXf quadratic::error(const MatrixXf & prediction, const RowVectorXi & classIndex) {
    MatrixXf err(prediction);
    for (unsigned n = 0; n < classIndex.size(); ++n) err(classIndex(n),n) -= 1;
    return err;
}


float quadratic::loss (const MatrixXf & prediction, const RowVectorXi & classIndex) {
    MatrixXf err(prediction);
    for (unsigned n = 0; n < classIndex.size(); ++n) err(classIndex(n),n) -= 1;
    float L = 0.5f*err.colwise().squaredNorm().mean();
    return L;
}



MatrixXf cross_entropy_softmax::error(const MatrixXf & prediction, const MatrixXf    & target) {
    MatrixXf Y = softmax_function(prediction);
    return Y-target;
}


float cross_entropy_softmax::loss (const MatrixXf & prediction, const MatrixXf    & target) {
    MatrixXf Y = softmax_function(prediction);
    Y = Y.array().log().matrix();
    float err = -(Y.cwiseProduct(target)).mean();
    err *= Y.rows();
    return err;
}



MatrixXf cross_entropy_softmax::error(const MatrixXf & prediction, const RowVectorXi    & classIndex) {
    MatrixXf grad = softmax_function(prediction);
    for (unsigned n = 0; n < classIndex.size(); ++n) grad(classIndex(n),n) -= 1;
    grad /= classIndex.size();
    return grad;
}



float cross_entropy_softmax::loss (const MatrixXf & prediction, const RowVectorXi    & classIndex) {
    MatrixXf grad = softmax_function(prediction);
    float err = 0;
    for (unsigned n = 0; n < classIndex.size(); ++n){
        err -= std::log(grad(classIndex(n),n));
    }
    err /= classIndex.size();
    return err;
}
