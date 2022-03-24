/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <vector>
#include <string>

#include "vector.h"
#include "real.h"
#include "dictionary.h"
#include "losslayer.h"


namespace fasttext {

class Sigmoid: public LossLayer{
public:
    Sigmoid(std::shared_ptr<Args>);
    ~Sigmoid();

    real loss(const std::vector<int32_t>& labels, real lr, Model *model_);
    void findKBest(int32_t top_k, real threshold, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_);
    real getLabelP(int32_t label, Vector &hidden, const Model *model_);
};

}