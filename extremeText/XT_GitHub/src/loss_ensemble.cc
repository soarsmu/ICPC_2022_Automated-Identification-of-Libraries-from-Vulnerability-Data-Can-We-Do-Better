/**
 * Copyright (c) 2018 by Marek Wydmuch, Róbert Busa-Fekete
 * All rights reserved.
 */

#include "loss_ensemble.h"

#include <random>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <vector>

#include "model.h"
#include "utils.h"

namespace fasttext {

Ensemble::Ensemble(std::shared_ptr<Args> args) : LossLayer(args){ }

Ensemble::~Ensemble(){ }

void Ensemble::setup(std::shared_ptr<Dictionary> dict, uint32_t seed){
    std::cerr << "Setting up Ensemble layer ...\n";
    rng.seed(seed);
    sizeSum = 0;
    args_->randomTree = true;

    assert(args_->ensemble > 0);
    for(auto i = 0; i < args_->ensemble; ++i){
        auto base = lossLayerFactory(args_, args_->loss);
        base->setup(dict, seed + i);
        base->setShift(sizeSum);
        sizeSum += base->getSize();
        baseLayers.push_back(base);
    }

    k = baseLayers[0]->getSize();
    multilabel = baseLayers[0]->isMultilabel();
}

real Ensemble::loss(const std::vector <int32_t> &input, const std::vector<int32_t>& labels, real lr, Model *model_){
    real lossSum = 0.0;
    real numOfUpdates = 0.0;
    std::string catInput = "&";

    if(args_->bagging < 1.0){
        for(auto i : input)
            catInput += "_" + std::to_string(i);
    }

    for(auto i = 0; i < baseLayers.size(); ++i) {
        if(args_->bagging < 1.0 && hashInput(std::to_string(i) + catInput) < args_->bagging) continue;
        lossSum += baseLayers[i]->loss(labels, lr, model_);
        numOfUpdates += 1.0;
    }

    return lossSum/args_->ensemble;
}

void Ensemble::findKBest(int32_t top_k, real threshold, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_){
    std::unordered_map <int32_t, real> label_freq;
    std::set<int32_t> label_set;

    for(int i=0; i < args_->ensemble; i++ ){
        heap.clear();
        baseLayers[i]->findKBest(top_k, 0.0, heap, hidden, model_);

        for(auto const& value: heap) {
            label_set.insert(value.second);
        }
    }

    heap.clear();    
    for(auto const& value : label_set) label_freq[value] = 0.0;
    
    for(int i=0; i < args_->ensemble; ++i){
        for(auto const& value : label_set){
            real prob = baseLayers[i]->getLabelP(value, hidden, model_);
            label_freq[value] += prob;
        }
    }

    for(const auto& elem: label_freq){
        real elem_prob = elem.second / ((real)args_->ensemble);
        if(elem_prob >= threshold) heap.push_back(std::make_pair(elem_prob, elem.first));
    }

    std::sort(heap.rbegin(), heap.rend());
    if(top_k < heap.size()) heap.resize(top_k);
}

int32_t Ensemble::getSize(){
    return sizeSum;
}

void Ensemble::save(std::ostream& out){
    std::cerr << "Saving Ensemble output ...\n";

    for(auto base : baseLayers)
        base->save(out);
}

void Ensemble::load(std::istream& in){
    std::cerr << "Loading Ensemble output ...\n";

    for(auto i = 0; i < args_->ensemble; ++i){
        auto base = lossLayerFactory(args_, args_->loss);
        base->load(in);
        baseLayers.push_back(base);
    }
}

real Ensemble::hashInput(const std::string& str){
    uint32_t h = utils::hash(str);
    uint32_t max = 1 << 24;
    return static_cast<real>(h % max) / max;
}

}
