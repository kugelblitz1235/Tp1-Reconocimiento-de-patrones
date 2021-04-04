#include "lossFunctions.h"
#include <math.h>

using namespace std;

LossFunctions::LossFunctions() {}

// Se asume que los vectores tienen la misma dimension
double LossFunctions::meanAbsoluteError(Vector& test, Vector& predicted){
    double res = 0;   
    
    for(unsigned int i = 0 ; i < test.rows() ; i++)
        res += abs(test(i,0) - predicted(i,0));

    return res/test.rows();
}

double LossFunctions::meanSquareError(Vector& test, Vector& predicted){
    double res = 0;   
    
    for(unsigned int i = 0 ; i < test.rows() ; i++)
        res += pow((test(i,0) - predicted(i,0)), 2);

    return res/test.rows();
}

double LossFunctions::rootMeanSquareError(Vector& test, Vector& predicted){
    double res = 0;   
    
    for(unsigned int i = 0 ; i < test.rows() ; i++)
        res += pow((test(i,0) - predicted(i,0)),2);

    
    return sqrt(res/test.rows());
}

double LossFunctions::rootMeanSquareLogError(Vector& test, Vector& predicted){
    double res = 0;   
    
    for(unsigned int i = 0 ; i < test.rows() ; i++)
        res += pow((log(test(i,0)+1) - log(predicted(i,0)+1)),2);
    
    return sqrt(res/test.rows());
}

// Aca hay un modulo para python que tiene estas loss functions
// From sklearn.metrics import mean_squared_error
