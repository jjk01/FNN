#include "loss.h"
#include <iostream>




MatrixXf quadratic::error(const MatrixXf & prediction, const MatrixXf & target) {
    return (prediction - target);
}


float quadratic::loss(const MatrixXf & prediction, const MatrixXf & target) {
    float L = 0.5*(prediction - target).colwise().squaredNorm().mean();
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
    float L = 0.5*err.colwise().squaredNorm().mean();
    return L;
}


/*
MatrixXf cross_entropy_with_softmax::error(const MatrixXf & y, const MatrixXf & y_true) {
    return y_true-y;
}


float cross_entropy_with_softmax::loss(const MatrixXf & y, const MatrixXf & y_true) {
    MatrixXf x = (y.array() - y.maxCoeff()).matrix();
    MatrixXf Y = (-x).array().exp().matrix();
    VectorXf norm = Y.colwise().sum().cwiseInverse();
    Y *= norm.asDiagonal();
    Y = Y.array().log().matrix();
    float err = -(y_true*Y).mean();
    return err;
}





MatrixXf cross_entropy_with_softmax_logits::error(const MatrixXf & y, const MatrixXf & logit) {
    MatrixXf err = -y;
    MatrixXi index = logit.cast<int>();
    for (unsigned n = 0; n < err.cols(); ++n) err(index(n),n) += 1;
    return err;
}


float cross_entropy_with_softmax_logits::loss(const MatrixXf & y, const MatrixXf & logit) {
    MatrixXf x = (y.array() - y.maxCoeff()).matrix();
    MatrixXf Y = (-x).array().exp().matrix();
    VectorXf norm = Y.colwise().sum().cwiseInverse();
    Y *= norm.asDiagonal();
    MatrixXi index = logit.cast<int>();

    float err = 0;
    for (unsigned n = 0; n < Y.cols(); ++n) err += std::log(Y(index(n),n));
    err /= Y.cols();

    return err;
}
*/
