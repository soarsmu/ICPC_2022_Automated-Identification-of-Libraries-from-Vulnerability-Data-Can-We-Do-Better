/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include "loss_sigmoid.h"

#include <algorithm>
#include <unordered_set>

#include "model.h"

namespace fasttext {

Sigmoid::Sigmoid(std::shared_ptr<Args> args) : LossLayer(args){
    multilabel = true;
}

Sigmoid::~Sigmoid() {

}

real Sigmoid::loss(const std::vector<int32_t>& labels, real lr, Model *model_) {

    real loss = 0.0;
    std::unordered_set<int32_t> positive_labels(labels.begin(), labels.end());

    for(int32_t i = 0; i < k; ++i){
        loss += binaryLogistic(i, positive_labels.count(i), lr, args_->l2, model_);
    }

    return loss;
}

void Sigmoid::findKBest(int32_t top_k, real threshold, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_) {

    for (int32_t i = 0; i < k; ++i) {
        real prob = getLabelP(i, hidden, model_);
        if (prob < threshold) continue;
        if (heap.size() == top_k && prob < heap.front().first) {
            continue;
        }
        heap.push_back(std::make_pair(prob, i));
        std::push_heap(heap.begin(), heap.end(), Model::comparePairs);
        if (heap.size() > top_k) {
            std::pop_heap(heap.begin(), heap.end(), Model::comparePairs);
            heap.pop_back();
        }
    }
}

real Sigmoid::getLabelP(int32_t label, Vector &hidden, const Model *model_){
    return model_->sigmoid((model_->quant_ && args_->qout)
                           ? model_->qwo_->dotRow(hidden, shift + label) : model_->wo_->dotRow(hidden, shift + label));
}

}