/**
 * Copyright (c) 2018 by Marek Wydmuch, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#pragma once

#include <iostream>
#include <ostream>
#include <vector>
#include <memory>

#include "real.h"
#include "vector.h"
#include "args.h"
#include "dictionary.h"

namespace fasttext {

class Model;

class LossLayer {
public:
    LossLayer(std::shared_ptr<Args>);
    virtual ~LossLayer();

    bool isMultilabel();
    int64_t getShift();
    void setShift(int64_t);
    void setSeed(uint32_t);

    virtual int32_t getSize();
    virtual void setup(std::shared_ptr<Dictionary>, uint32_t);

    real binaryLogistic(int32_t, real, real, real, Model*);

    virtual real loss(const int32_t target, real lr, Model *model_);
    virtual real loss(const std::vector <int32_t> &labels, real lr, Model *model_);
    virtual real loss(const std::vector <int32_t> &input, const std::vector <int32_t> &labels, real lr, Model *model_);
    virtual void findKBest(int32_t top_k, real threshold, std::vector <std::pair<real, int32_t>> &heap, Vector &hidden, const Model *model_) = 0;
    virtual real getLabelP(int32_t label, Vector &hidden, const Model *model_);

    virtual void save(std::ostream&);
    virtual void load(std::istream&);

    virtual void printInfo();

protected:
    std::default_random_engine rng;
    std::shared_ptr<Args> args_;

    int32_t k; // number of classes/labels
    int64_t shift;
    bool multilabel;
};

std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args);
std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args, loss_name loss);

}
