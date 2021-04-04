#include <algorithm>
//#include <chrono>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;

LinearRegression::LinearRegression() {}

void LinearRegression::fit(Matrix X, Matrix y) {
    _x_solucion = LinearRegression :: normalEquations(X,y);
}

void LinearRegression::fitWithSVD(Matrix X, Matrix y) {
    _x_solucion = LinearRegression :: svdDescomposition(X,y);
    
}

void LinearRegression::fitWithQR(Matrix X, Matrix y) {
    _x_solucion = LinearRegression :: qrDescomposition(X,y);
}

void LinearRegression::fitWithNormalEq(Matrix X, Matrix y) {
    _x_solucion = LinearRegression :: normalEquationsWithEigen(X,y);
}

Matrix LinearRegression :: normalEquations(Matrix X, Matrix y) {
    Matrix AtA = X.transpose() * X;
    Matrix Atb = X.transpose() * y;

    return AtA.inverse() * Atb;
}

Matrix LinearRegression::normalEquationsWithEigen(Matrix X, Matrix y) {
    return (X.transpose() * X).ldlt().solve(X.transpose() * y);
}

Matrix LinearRegression ::svdDescomposition(Matrix X, Matrix y) {
    return X.bdcSvd(Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV).solve(y);
}

Matrix LinearRegression ::qrDescomposition (Matrix X, Matrix y) {
    return X.colPivHouseholderQr().solve(y); 
}

Matrix LinearRegression::predict(Matrix X) {
    return X * _x_solucion;
}
