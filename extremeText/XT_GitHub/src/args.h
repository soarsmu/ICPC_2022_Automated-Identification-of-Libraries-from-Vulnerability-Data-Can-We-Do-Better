/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "real.h"

namespace fasttext {

enum class model_name : int { cbow = 1, sg, sup };
enum class loss_name : int { hs = 1, ns, softmax, plt, sigmoid };
enum class tree_type_name : int { huffman = 1, complete, kmeans, custom };

class Args {
  protected:
    std::string lossToString(loss_name) const;
    std::string boolToString(bool) const;
    std::string modelToString(model_name) const;
    std::string treeTypeToString(tree_type_name) const;

  public:
    Args();
    int seed;
    bool train;
    std::string input;
    std::string output;
    double lr;
    int lrUpdateRate;
    int dim;
    int ws;
    int epoch;
    int minCount;
    int minCountLabel;
    int neg;
    int wordNgrams;
    loss_name loss;
    model_name model;
    int bucket;
    int minn;
    int maxn;
    int thread;
    double t;
    std::string label;
    int verbose;
    bool saveOutput;
    bool saveVectors;
    bool saveDocuments;

    // Vectors init
    std::string pretrainedVectors;
    bool freezeVectors;
    bool initZeros;

    // Features args
    bool wordsWeights;
    bool tfidfWeights;
    real weightsThr;
    bool addEosToken;
    real eosWeight;
    std::string weight;
    std::string tag;

    // Quantization args
    bool qout;
    bool retrain;
    bool qnorm;
    size_t cutoff;
    size_t dsub;

    // PLT args
    int arity;
    bool probNorm;
    tree_type_name treeType;
    std::string treeStructure;
    bool randomTree;
    int maxLeaves;

    // K-means
    real kMeansEps;
    bool kMeansBalanced;
    real kMeansCentThr;
    real kMeansSample;

    // Update args
    real l2;
    real lrDecay;

    // Ensemble args
    int ensemble;
    real bagging;

    void parseArgs(const std::vector<std::string>& args);
    void printHelp();
    void printBasicHelp();
    void printDictionaryHelp();
    void printTrainingHelp();
    void printQuantizationHelp();
    void printInfo();
    void save(std::ostream&);
    void load(std::istream&);
    void dump(std::ostream&) const;
};
}
