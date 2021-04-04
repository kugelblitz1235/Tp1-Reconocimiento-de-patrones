#pragma once

#include "types.h"

#include <vector>

class LossFunctions {
    public:
        LossFunctions();

        double meanAbsoluteError(Vector &test, Vector &ppredicted);

        double meanSquareError(Vector &test, Vector &ppredicted);

        double rootMeanSquareError(Vector &test, Vector &ppredicted);

        double rootMeanSquareLogError(Vector &test, Vector &ppredicted);

    private:
};