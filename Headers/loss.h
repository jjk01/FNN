#ifndef LOSS_H
#define LOSS_H

#include <Eigen/Dense>
#include "exception.h"

using namespace Eigen;



enum class LossType {
    quadratic,
    cross_entropy_softmax,
};



struct Loss {
    virtual ~Loss() = default;

    virtual MatrixXf error(const MatrixXf & prediction,  const MatrixXf    & target) = 0;
    virtual float    loss (const MatrixXf & prediction,  const MatrixXf    & target) = 0;

    virtual MatrixXf error(const MatrixXf  & prediction, const RowVectorXi    & classIndex) = 0;
    virtual float     loss (const MatrixXf & prediction, const RowVectorXi    & classIndex) = 0;
};


struct quadratic: public Loss {
    MatrixXf error(const MatrixXf & prediction, const MatrixXf    & target);
    float    loss (const MatrixXf & prediction, const MatrixXf    & target);

    MatrixXf error(const MatrixXf & prediction, const RowVectorXi    & classIndex);
    float    loss (const MatrixXf & prediction, const RowVectorXi    & classIndex);
};


struct cross_entropy_softmax: public Loss {
    MatrixXf error(const MatrixXf & prediction, const MatrixXf    & target);
    float    loss (const MatrixXf & prediction, const MatrixXf    & target);

    MatrixXf error(const MatrixXf & prediction, const RowVectorXi    & classIndex);
    float    loss (const MatrixXf & prediction, const RowVectorXi    & classIndex);
};

#endif /* LOSS_H */
