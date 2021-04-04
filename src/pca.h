#pragma once
#include "types.h"

class PCA {
    public:
        PCA(unsigned int components);

        void fit(Matrix X);

        Eigen::MatrixXd transform(Matrix X);
        Eigen::MatrixXd getEigenvectorMatrix(Matrix X);
        Eigen::VectorXd getEigenvaluesMatrix(Matrix X);
    private:
        Matrix _V;
        unsigned int _alpha;
};
