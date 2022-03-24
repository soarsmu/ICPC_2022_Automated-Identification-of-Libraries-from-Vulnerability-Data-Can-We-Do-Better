/**
 * Copyright (c) 2018 by Marek Wydmuch, Róbert Busa-Fekete
 * All rights reserved.
 */

#pragma once

#include "losslayer.h"
#include <random>

namespace fasttext {
    
class Ensemble: public LossLayer{
public:
    Ensemble(std::shared_ptr<Args>);
    ~Ensemble();

    int32_t getSize();
    void setup(std::shared_ptr<Dictionary>, uint32_t seed);

    real loss(const std::vector <int32_t> &input, const std::vector<int32_t>& labels, real lr, Model *model_);
    void findKBest(int32_t top_k, real threshold, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_);

    void save(std::ostream&);
    void load(std::istream&);

private:
    std::vector<std::shared_ptr<LossLayer>> baseLayers;
    int64_t sizeSum;

    real hashInput(const std::string& input);
};

}
