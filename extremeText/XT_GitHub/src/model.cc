/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace fasttext {

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Args> args,
    std::shared_ptr<LossLayer> lossLayer,
    int32_t seed)
    : hidden_(args->dim),
      output_(wo->size(0)),
      grad_(args->dim),
      rng(seed),
      quant_(false) {
  wi_ = wi;
  wo_ = wo;
  args_ = args;
  lossLayer_ = lossLayer;
  osz_ = wo->size(0);
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  t_log_.reserve(LOG_TABLE_SIZE + 1);
  initSigmoid();
  initLog();
}

void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi,
                               std::shared_ptr<QMatrix> qwo, bool qout) {
  qwi_ = qwi;
  qwo_ = qwo;
  if (qout) {
    osz_ = qwo_->getM();
  }
}

real Model::binaryLogistic(int32_t target, bool label, real lr) {
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real diff = real(label) - score;

  // Original update
  /*
  real alpha = lr * (real(label) - score);
  grad_.addRow(*wo_, target, alpha);
  wo_->addRow(hidden_, target, alpha);
  */

  grad_.addRowL2(*wo_, target, lr, diff, args_->l2);
  wo_->addRowL2(hidden_, target, lr, diff, args_->l2);

  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::negativeSampling(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr);
    } else {
      loss += binaryLogistic(getNegative(target), false, lr);
    }
  }
  return loss;
}

real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  if (quant_ && args_->qout) {
    output.mul(*qwo_, hidden);
  } else {
    output.mul(*wo_, hidden);
  }
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}

real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]);
}

void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    if(quant_) {
      hidden.addRow(*qwi_, *it);
    } else {
      hidden.addRow(*wi_, *it);
    }
  }
  hidden.mul(1.0 / input.size());
}

real Model::computeHidden(const std::vector<int32_t>& input, const std::vector<real>& input_values, Vector& hidden) const {
  assert(hidden.size() == hsz_);
  assert(input.size() == input_values.size());

  hidden.zero();
  real values_sum = 0;
  for (auto it = 0; it < input.size(); ++it){
    hidden.addRow(*wi_, input[it], input_values[it]);
    values_sum += input_values[it];
  }
  hidden.mul(1.0 / values_sum);

  //assert(input.size() == values_sum);
  return values_sum;
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r) {
  return l.first > r.first;
}

void Model::predict(int32_t k, real threshold, std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  if (args_->model != model_name::sup) {
    throw std::invalid_argument("Model needs to be supervised for prediction!");
  }

  if (lossLayer_ != nullptr){
    lossLayer_->findKBest(k, threshold, heap, hidden, this);
  }
  else if (args_->loss == loss_name::hs) {
    dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, hidden);
  } else {
    findKBest(k, threshold, heap, hidden, output);
  }
  std::sort(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(const std::vector<int32_t>& input, const std::vector<real>& input_values,
                    int32_t k, real threshold, std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  heap.reserve(k + 1);
  computeHidden(input, input_values, hidden);
  predict(k, threshold, heap, hidden, output);
}

void Model::predict(
  const std::vector<int32_t>& input,
  const std::vector<real>& input_values,
  int32_t k,
  real threshold,
  std::vector<std::pair<real, int32_t>>& heap
) {
  predict(input, input_values, k, threshold, heap, hidden_, output_);
}

void Model::findKBest(
  int32_t k,
  real threshold,
  std::vector<std::pair<real, int32_t>>& heap,
  Vector& hidden, Vector& output
) const {
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    if (output[i] < threshold) continue;
    if (heap.size() == k && std_log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(std_log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

real Model::getProb(Vector& hidden, int32_t target){
  if (lossLayer_ != nullptr){
    real p = lossLayer_->getLabelP(target, hidden, this);
    return p;
  } else throw  std::invalid_argument("Only lossLayers support getProb!");
}

real Model::getProb(const std::vector<int32_t>& input, const std::vector<real>& input_values, Vector& hidden, int32_t target){
  if (input.size() == 0) return 0;
  computeHidden(input, input_values, hidden);
  if (lossLayer_ != nullptr){
    real p = lossLayer_->getLabelP(target, hidden, this);
    return p;
  } else throw  std::invalid_argument("Only lossLayers support getProb!");
}

void Model::dfs(int32_t k, real threshold, int32_t node, real score,
                std::vector<std::pair<real, int32_t>>& heap,
                Vector& hidden) const {
  if (score < std_log(threshold)) return;
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree[node].left == -1 && tree[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f;
  if (quant_ && args_->qout) {
    f= qwo_->dotRow(hidden, node - osz_);
  } else {
    f= wo_->dotRow(hidden, node - osz_);
  }
  f = 1. / (1 + std::exp(-f));

  dfs(k, threshold, tree[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree[node].right, score + std_log(f), heap, hidden);
}

void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;
  computeHidden(input, hidden_);
  if (args_->loss == loss_name::ns) {
    loss_ += negativeSampling(target, lr);
  } else if (args_->loss == loss_name::hs) {
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
    loss_ += softmax(target, lr);
  }
  nexamples_ += 1;

  if (args_->model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addRow(grad_, *it, 1.0);
  }
}

void Model::update(const std::vector<int32_t>& input, const std::vector<real>& input_values, const std::vector<int32_t>& labels, real lr) {

  if (input.size() == 0) return;
  real values_sum = computeHidden(input, input_values, hidden_);
  grad_.zero();

  if (lossLayer_ != nullptr) {
    loss_ += lossLayer_->loss(input, labels, lr, this);
  } else {
    std::uniform_int_distribution<> uniform(0, labels.size() - 1);
    int32_t target = labels[uniform(rng)];
    assert(target >= 0);
    assert(target < osz_);

    if (args_->loss == loss_name::ns) {
      loss_ += negativeSampling(target, lr);
    } else if (args_->loss == loss_name::hs) {
      loss_ += hierarchicalSoftmax(target, lr);
    } else {
      loss_ += softmax(target, lr);
    }
  }

  nexamples_ += 1;
  if (!args_->freezeVectors) {
    if (args_->model == model_name::sup) {
      grad_.mul(1.0 / values_sum);
    }
    for (auto it = 0; it < input.size(); ++it) {
      wi_->addRow(grad_, input[it], input_values[it]);
    }
  }
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs) {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives_.push_back(i);
    }
  }
  std::shuffle(negatives_.begin(), negatives_.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives_[negpos];
    negpos = (negpos + 1) % negatives_.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& counts) {
  if(args_->verbose > 2)
      std::cerr << "  Building HS tree ...\n";

  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const {
  return loss_ / nexamples_;
}

void Model::initSigmoid() {
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }
}

void Model::initLog() {
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Model::log(real x) const {
  //return std::log(x+1e-5);

  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Model::std_log(real x) const {
  return std::log(x+1e-5);
}

real Model::sigmoid(real x) const {
  //return 1.0 / (1.0 + std::exp(-x));

  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}

}
