/**
 * Copyright (c) 2018 by Marek Wydmuch, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#include "losslayer.h"
#include "model.h"
#include "loss_plt.h"
#include "loss_sigmoid.h"
#include "loss_ensemble.h"

namespace fasttext {

std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args, loss_name loss){
    if (loss == loss_name::plt)
        return std::static_pointer_cast<LossLayer>(std::make_shared<PLT>(args));
    if (loss == loss_name::sigmoid)
        return std::static_pointer_cast<LossLayer>(std::make_shared<Sigmoid>(args));

    return nullptr;

    // TODO: Wrap vanilla fastText losses as LossLayer classes
    //std::cerr << "Unknown loss type!\n";
    //exit(1);
}

std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args){
    if(args->ensemble > 1)
        return std::static_pointer_cast<LossLayer>(std::make_shared<Ensemble>(args));
    else
        return lossLayerFactory(args, args->loss);
}

LossLayer::LossLayer(std::shared_ptr<Args> args){
    args_ = args;
    multilabel = false;
    shift = 0;
}

LossLayer::~LossLayer(){

}

int32_t LossLayer::getSize(){
    return k;
}

void LossLayer::setup(std::shared_ptr<Dictionary> dict, uint32_t seed){
    rng.seed(seed);
    k = dict->nlabels();
}

real LossLayer::binaryLogistic(int32_t target, real label, real lr, real l2, Model *model_){
    real score = model_->sigmoid(model_->wo_->dotRow(model_->hidden_, shift + target));
    real diff = (label - score);

    model_->grad_.addRowL2(*model_->wo_, shift + target, lr, diff / args_->ensemble, l2);
    model_->wo_->addRowL2(model_->hidden_, shift + target, lr, diff, l2);

    if (label) {
        return -model_->log(score);
    } else {
        return -model_->log(1.0 - score);
    }
}

real LossLayer::loss(const int32_t target, real lr, Model *model_){
    if(multilabel){
        std::vector <int32_t> target_ = {target};
        return loss(target_, lr, model_);
    }

    std::cerr << "LossLayer doesn't have loss function!\n";
    return 0;
}

real LossLayer::loss(const std::vector <int32_t> &labels, real lr, Model *model_){
    // Pick one label heuristic
    if(!multilabel){
        std::uniform_int_distribution<> uniform(0, labels.size() - 1);
        int32_t target_ = labels[uniform(rng)];
        return loss(target_, lr, model_);
    }

    std::cerr << "LossLayer doesn't have loss function!\n";
    return 0;
}

real LossLayer::loss(const std::vector <int32_t> &input, const std::vector <int32_t> &labels, real lr, Model *model_){
    return loss(labels, lr, model_);
}

// TODO: Make it abstract
real LossLayer::getLabelP(int32_t label, Vector &hidden, const Model *model_){
    return 0;
}

bool LossLayer::isMultilabel(){
    return multilabel;
}

void LossLayer::setShift(int64_t shift_){
    shift = shift_;
}

int64_t LossLayer::getShift(){
    return shift;
}

void LossLayer::setSeed(uint32_t seed_){
    rng.seed(seed_);
}

void LossLayer::printInfo(){

}

void LossLayer::save(std::ostream& out){
    out.write((char*) &k, sizeof(int32_t));
    out.write((char*) &shift, sizeof(shift));
}

void LossLayer::load(std::istream& in){
    in.read((char*) &k, sizeof(int32_t));
    in.read((char*) &shift, sizeof(shift));
}

}
