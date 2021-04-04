#pragma once

#include "types.h"

class LinearRegression {
    public:
        LinearRegression();

        void fit(Matrix X, Matrix y);

        void fitWithSVD(Matrix X, Matrix y);

        void fitWithQR(Matrix X, Matrix y);
        
        void fitWithNormalEq(Matrix X, Matrix y);

        Matrix normalEquations(Matrix X, Matrix y);
        
        Matrix svdDescomposition(Matrix X, Matrix y);
        
        Matrix qrDescomposition (Matrix X, Matrix y);
        
        Matrix normalEquationsWithEigen (Matrix X, Matrix y);

        Matrix predict(Matrix X);
        
    private:
        Matrix _x_solucion;
};
