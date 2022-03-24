/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * Copyright (c) 2018 by Marek Wydmuch, Róbert Busa-Fekete
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"
#include "threads.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>


namespace fasttext {

constexpr int32_t FASTTEXT_VERSION = 14102; /* extremeText Version E1.0.2 */
constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 14102;

FastText::FastText() : quant_(false) {}

void FastText::addInputVector(Vector& vec, int32_t ind) const {
  if (quant_) {
    vec.addRow(*qinput_, ind);
  } else {
    vec.addRow(*input_, ind);
  }
}

std::shared_ptr<const Dictionary> FastText::getDictionary() const {
  return dict_;
}

const Args FastText::getArgs() const {
  return *args_.get();
}

std::shared_ptr<const Matrix> FastText::getInputMatrix() const {
  return input_;
}

std::shared_ptr<const Matrix> FastText::getOutputMatrix() const {
  return output_;
}

int32_t FastText::getWordId(const std::string& word) const {
  return dict_->getId(word);
}

int32_t FastText::getSubwordId(const std::string& word) const {
  int32_t h = utils::hash(word) % args_->bucket;
  return dict_->nwords() + h;
}

void FastText::getWordVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
  vec.zero();
  for (int i = 0; i < ngrams.size(); i ++) {
    addInputVector(vec, ngrams[i]);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::getVector(Vector& vec, const std::string& word) const {
  getWordVector(vec, word);
}

void FastText::getSubwordVector(Vector& vec, const std::string& subword)
    const {
  vec.zero();
  int32_t h = utils::hash(subword) % args_->bucket;
  h = h + dict_->nwords();
  addInputVector(vec, h);
}

void FastText::saveVectors() {
  if (args_->verbose > 2)
    std::cerr << "Saving vectors ...\n";

  std::ofstream ofs(args_->output + ".vec");
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        args_->output + ".vec" + " cannot be opened for saving vectors!");
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput() {
  if (args_->verbose > 2)
    std::cerr << "Saving output ...\n";

  std::ofstream ofs(args_->output + ".output");
  if (!ofs.is_open()) {
    throw std::invalid_argument(
        args_->output + ".output" + " cannot be opened for saving vectors!");
  }
  if (quant_) {
    throw std::invalid_argument(
        "Option -saveOutput is not supported for quantized models.");
  }
  int32_t n = (args_->model == model_name::sup) ? dict_->nlabels()
                                                : dict_->nwords();
  ofs << n << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < n; i++) {
    std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                         : dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveDocuments(const std::string input, const std::string output, const std::string suffix) {
  if (args_->verbose > 2)
    std::cerr << "Saving documents ...\n";

  if (args_->model != model_name::sup){
    throw std::invalid_argument(
            "Option -saveDocuments is not supported for not supervised models.");
  }

  std::string output_ = output.substr(0, output.size() - 4) + "_" + suffix + ".txt";
  std::ifstream ifs(input);
  std::ofstream ofs(output_);
  if (!ofs.is_open()) {
    throw std::invalid_argument(
            output_ + " cannot be opened for saving vectors!");
  }

  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;
  Vector hidden_(args_->dim);

  uint32_t docs = 0;
  std::string dummyLine;
  while(std::getline(ifs, dummyLine)) ++docs;
  ifs.close();
  ifs.open(input);

  ofs << docs << " " << args_->dim << " " << dict_->nlabels() << "\n";

  for (int32_t i = 0; i < docs; i++) {
    dict_->getLine(ifs, line, line_values, labels, tags);
    model_->computeHidden(line, line_values, hidden_);

    if(labels.size()){
      ofs << labels[0];
      for (int32_t l = 1; l < labels.size(); ++l) ofs << "," << labels[l];
    }
    for (int32_t f = 0; f < args_->dim; ++f) ofs << " " << f << ":" << hidden_[f];
    ofs << "\n";
  }
  ofs.close();
  ifs.close();
}

bool FastText::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void FastText::signModel(std::ostream& out) {
  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void FastText::saveModel() {
  std::string fn(args_->output);
  if (quant_) {
    fn += ".ftz";
  } else {
    fn += ".bin";
  }
  saveModel(fn);
}

void FastText::saveModel(const std::string path) {
  if (args_->verbose > 2)
    std::cerr << "Saving model ...\n";

  std::ofstream ofs(path, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::invalid_argument(path + " cannot be opened for saving!");
  }
  signModel(ofs);
  args_->save(ofs);
  dict_->save(ofs);

  ofs.write((char*)&(quant_), sizeof(bool));
  if (quant_) {
    qinput_->save(ofs);
  } else {
    input_->save(ofs);
  }

  ofs.write((char*)&(args_->qout), sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->save(ofs);
  } else {
    output_->save(ofs);
  }

  if(lossLayer_ != nullptr)
    lossLayer_->save(ofs);

  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::cerr << "Loading model ...\n";

  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  qinput_ = std::make_shared<QMatrix>();
  qoutput_ = std::make_shared<QMatrix>();
  args_->load(in);
  if (version == 11 && args_->model == model_name::sup) {
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
  }
  dict_ = std::make_shared<Dictionary>(args_, in);

  bool quant_input;
  in.read((char*) &quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    qinput_->load(in);
  } else {
    input_->load(in);
  }

  if (!quant_input && dict_->isPruned()) {
    throw std::invalid_argument(
        "Invalid model file.\n"
        "Please download the updated model from www.fasttext.cc.\n"
        "See issue #332 on Github for more information.\n");
  }

  in.read((char*) &args_->qout, sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->load(in);
  } else {
    output_->load(in);
  }

  lossLayer_ = lossLayerFactory(args_);
  if(lossLayer_ != nullptr)
    lossLayer_->load(in);

  model_ = std::make_shared<Model>(input_, output_, args_, lossLayer_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);

  if(lossLayer_ == nullptr){
    if (args_->model == model_name::sup) {
      model_->setTargetCounts(dict_->getCounts(entry_type::label));
    } else {
      model_->setTargetCounts(dict_->getCounts(entry_type::word));
    }
  }
}

void FastText::printInfo(real progress, real loss, std::ostream& log_stream) {
  // clock_t might also only be 32bits wide on some systems
  double t = double(clock() - start_) / double(CLOCKS_PER_SEC);
  double lr = args_->lr * (args_->lrDecay != 1.0
                           ? std::pow((1.0 - std::pow(progress, args_->lrDecay)), 1.0/args_->lrDecay)
                           : 1.0 - progress);
  double wst = 0;
  int64_t eta = 720 * 3600; // Default to one month
  if (progress > 0 && t >= 0) {
    eta = int(t / progress * (1 - progress) / args_->thread);
    wst = double(tokenCount_) / t;
  }
  int32_t etah = eta / 3600;
  int32_t etam = (eta % 3600) / 60;
  progress = progress * 100;
  log_stream << std::fixed;
  log_stream << "Progress: ";
  log_stream << std::setprecision(1) << std::setw(5) << progress << "%";
  log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
  log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
  log_stream << " loss: " << std::setw(9) << std::setprecision(6) << loss;
  log_stream << " ETA: " << std::setw(3) << etah;
  log_stream << "h" << std::setw(2) << etam << "m";
  log_stream << std::flush;
}

std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const {
  Vector norms(input_->size(0));
  input_->l2NormRow(norms);
  std::vector<int32_t> idx(input_->size(0), 0);
  std::iota(idx.begin(), idx.end(), 0);
  auto eosid = dict_->getId(Dictionary::EOS);
  std::sort(idx.begin(), idx.end(),
      [&norms, eosid] (size_t i1, size_t i2) {
      return eosid ==i1 || (eosid != i2 && norms[i1] > norms[i2]);
      });
  idx.erase(idx.begin() + cutoff, idx.end());
  return idx;
}

void FastText::quantize(const Args qargs) {
  if (args_->model != model_name::sup) {
    throw std::invalid_argument(
        "For now we only support quantization of supervised models");
  }
  args_->input = qargs.input;
  args_->qout = qargs.qout;
  args_->output = qargs.output;

  if (qargs.cutoff > 0 && qargs.cutoff < input_->size(0)) {
    auto idx = selectEmbeddings(qargs.cutoff);
    dict_->prune(idx);
    std::shared_ptr<Matrix> ninput =
        std::make_shared<Matrix>(idx.size(), args_->dim);
    for (auto i = 0; i < idx.size(); i++) {
      for (auto j = 0; j < args_->dim; j++) {
        ninput->at(i, j) = input_->at(idx[i], j);
      }
    }
    input_ = ninput;
    if (qargs.retrain) {
      args_->epoch = qargs.epoch;
      args_->lr = qargs.lr;
      args_->thread = qargs.thread;
      args_->verbose = qargs.verbose;
      startThreads();
    }
  }

  qinput_ = std::make_shared<QMatrix>(*input_, qargs.dsub, qargs.qnorm);

  if (args_->qout) {
    qoutput_ = std::make_shared<QMatrix>(*output_, 2, qargs.qnorm);
  }

  quant_ = true;
  model_ = std::make_shared<Model>(input_, output_, args_, nullptr, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);
  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::supervised(
    Model& model,
    real lr,
    const std::vector<int32_t>& line,
    const std::vector<real>& line_values,
    const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  model.update(line, line_values, labels, lr);
}

void FastText::cbow(Model& model, real lr,
                    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

std::tuple<uint64_t, double, double, double> FastText::testInThreads(
        std::string infile,
        int32_t k,
        int32_t thread,
        real threshold) {

  if (args_->verbose > 0) {
    std::cerr << "Test in " << thread << " threads ...\n"
              << "  K: " << k << ", threshold: " << threshold << std::endl;
    args_->printInfo();
  }

  TestThreadResult testResults;
  testResults.nexamples = 0;
  testResults.precision = 0;
  testResults.npredictions = 0;
  testResults.nlabels = 0;
  TestThreadResult* testResultPtr = &testResults;

  std::vector<std::thread> threads;
  std::mutex testMutex;
  std::mutex* testMutexPtr = &testMutex;
  for (int32_t i = 0; i < thread; i++)
    threads.push_back(std::thread([=]() { testThread(i, thread, infile, k, threshold, testResultPtr, testMutexPtr); }));
    //threads.push_back(std::thread([=]() { testThread(i, thread, infile, k, threshold, &testResults, &testMutex); }));
  for (int32_t i = 0; i < thread; i++) threads[i].join();

  return std::tuple<uint64_t, double, double, double>(
          testResults.nexamples, testResults.precision / testResults.npredictions,
          testResults.precision / testResults.nlabels, static_cast<double>(testResults.coverage.size()) / dict_->nlabels());
}

void FastText::testThread(
        int32_t threadId,
        int32_t thread,
        std::string infile,
        int32_t k,
        real threshold,
        TestThreadResult* testResults,
        std::mutex* testMutex) {

  std::ifstream ifs(infile);
  int64_t insize = utils::size(ifs);
  utils::seek(ifs, threadId * insize / thread);
  int64_t startpos = ifs.tellg();
  int64_t endpos = (threadId + 1) * insize / thread;

  uint64_t nexamples = 0, nlabels = 0, npredictions = 0;
  double precision = 0.0;
  std::unordered_set<int32_t> coverage;

  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;

  std::vector<std::pair<real,int32_t>> modelPredictions;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());

  while (ifs.peek() != EOF) {
    dict_->getLine(ifs, line, line_values, labels, tags);
    auto pos = ifs.tellg();
    if(threadId == 0 && args_->verbose > 0) utils::printProgress(startpos, pos, endpos, std::cerr);
    if(pos < startpos || pos > endpos) break;

    if (labels.size() > 0 && line.size() > 0) {
      modelPredictions.clear();
      model_->computeHidden(line, line_values, hidden);
      model_->predict(k, threshold, modelPredictions, hidden, output);

      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
          coverage.insert(it->second);
        }
      }
      nexamples++;
      nlabels += labels.size();
      npredictions += modelPredictions.size();
    }
  }

  ifs.close();

  testMutex->lock();
  testResults->nexamples += nexamples;
  testResults->precision += precision;
  testResults->npredictions += npredictions;
  testResults->nlabels += nlabels;
  for(auto& l : coverage) testResults->coverage.insert(l);
  testMutex->unlock();
}

std::tuple<uint64_t, double, double, double> FastText::test(
    std::istream& in,
    int32_t k,
    real threshold) {

  if (args_->verbose > 0) {
    std::cerr << "Test ...\n" << "  K: " << k << ", threshold: " << threshold << std::endl;
    args_->printInfo();
  }

  uint64_t nexamples = 0, nlabels = 0, npredictions = 0;
  double precision = 0.0;
  std::unordered_set<int32_t> coverage;
  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;
  while (in.peek() != EOF) {
    dict_->getLine(in, line, line_values, labels, tags);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, line_values, k, threshold, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
          coverage.insert(it->second);
        }
      }
      nexamples++;
      nlabels += labels.size();
      npredictions += modelPredictions.size();
    }
  }
  return std::tuple<uint64_t, double, double, double>(
      nexamples, precision / npredictions, precision / nlabels, static_cast<double>(coverage.size())/ dict_->nlabels());
}

void FastText::predictInThreads(
  std::string infile,
  std::string outfile,
  int32_t thread,
  int32_t k,
  bool print_prob,
  real threshold){

  if (args_->verbose > 0) {
  std::cerr << "Predict in " << thread << " threads ...\n"
            << "  K: " << k << ", threshold: " << threshold << std::endl;
    args_->printInfo();
  }

  std::vector<std::thread> threads;
  for (int32_t i = 0; i < thread; i++)
    threads.push_back(std::thread([=]() { predictThread(i, thread, infile, outfile, k, print_prob, threshold); }));
  for (int32_t i = 0; i < thread; i++) threads[i].join();
}

void FastText::predictThread(
    int32_t threadId,
    int32_t thread,
    std::string infile,
    std::string outfile,
    int32_t k,
    bool print_prob,
    real threshold) {
  std::ifstream ifs(infile);
  std::ofstream ofs(outfile + "." + utils::itos(threadId, 3));

  int64_t insize = utils::size(ifs);
  utils::seek(ifs, threadId * insize / thread);
  int64_t startpos = ifs.tellg();
  int64_t endpos = (threadId + 1) * insize / thread;

  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;

  std::vector<std::pair<real,int32_t>> modelPredictions;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());

  while (ifs.peek() != EOF) {
    dict_->getLine(ifs, line, line_values, labels, tags);
    auto pos = ifs.tellg();
    if(threadId == 0 && args_->verbose > 0) utils::printProgress(startpos, pos, endpos, std::cerr);
    if(pos < startpos || pos > endpos) break;

    modelPredictions.clear();
    if (line.empty()) continue;

    model_->computeHidden(line, line_values, hidden);
    model_->predict(k, threshold, modelPredictions, hidden, output);

    for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {

      if (it != modelPredictions.cbegin()) ofs << " ";
      ofs << dict_->getLabel(it->second);
      if (print_prob) ofs << " " << it->first;
    }

    for (auto &t : tags) ofs << " " << t;
    ofs << std::endl;
  }

  ifs.close();
  ofs.close();
}

void FastText::predict(
  std::istream& in,
  int32_t k,
  std::vector<std::pair<real,std::string>>& predictions,
  real threshold
) const {
  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;
  dict_->getLine(in, line, line_values, labels, tags);

  predictions.clear();
  if (line.empty()) return;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());
  std::vector<std::pair<real,int32_t>> modelPredictions;
  model_->predict(line, line_values, k, threshold, modelPredictions, hidden, output);
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
    predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::predict(
  std::istream& in,
  int32_t k,
  bool print_prob,
  real threshold
) {

  if (args_->verbose > 0) {
    std::cerr << "Predict ...\n" << "  k: " << k << ", threshold: " << threshold << std::endl;
    args_->printInfo();
  }

  std::vector<std::pair<real,std::string>> predictions;
  while (in.peek() != EOF) {
    predictions.clear();
    predict(in, k, predictions, threshold);
    if (predictions.empty()) {
      std::cout << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << " ";
      }
      std::cout << it->second;
      if (print_prob) {
          std::cout << " " << it->first;
      }
    }
    std::cout << std::endl;
  }
}

void FastText::outputProb(std::ostream& ofs, Vector& hidden, std::vector<int32_t>& labels, std::vector<std::string>& tags, real threshold){
  int out_count = 0;
  for (auto l = labels.cbegin(); l != labels.cend(); l++) {
    real p = model_->getProb(hidden, *l);
    if (p >= threshold) {
      if (out_count) ofs << " ";
      ofs << dict_->getLabel(*l) << " " << p;
      ++out_count;
    }
  }

  for (auto &t : tags) ofs << " " << t;
  ofs << std::endl;
}

void FastText::getProbInThreads(std::string infile, std::string outfile, int32_t thread, real threshold){
  if (args_->verbose > 0) {
    std::cerr << "Getting probabilities " << args_->thread << " threads ...\n"
              << "  Threshold: " << threshold << std::endl;
    args_->printInfo();
  }

  std::vector<std::thread> threads;
  for (int32_t i = 0; i < thread; i++)
    threads.push_back(std::thread([=]() { getProbThread(i, thread, infile, outfile, threshold); }));
  for (int32_t i = 0; i < thread; i++)
    threads[i].join();
}

void FastText::getProbThread(int32_t threadId, int32_t thread, std::string infile, std::string outfile, real threshold) {
  std::ifstream ifs(infile);
  std::ofstream ofs(outfile + "." + utils::itos(threadId, 3));

  int64_t insize = utils::size(ifs);
  utils::seek(ifs, threadId * insize / thread);
  int64_t startpos = ifs.tellg();
  int64_t endpos = (threadId + 1) * insize / thread;

  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;
  Vector hidden(args_->dim);
  while (ifs.peek() != EOF) {
    dict_->getLine(ifs, line, line_values, labels, tags);
    if(ifs.tellg() < startpos || ifs.tellg() > endpos) break;

    model_->computeHidden(line, line_values, hidden);
    outputProb(ofs, hidden, labels, tags, threshold);
  }

  ifs.close();
  ofs.close();
}

void FastText::getProb(std::istream& in, real threshold) {
  if (args_->verbose > 0) {
    std::cerr << "Getting probabilities ...\n"
              << "  Threshold: " << threshold << std::endl;
    args_->printInfo();
  }

  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;
  Vector hidden(args_->dim);
  while (in.peek() != EOF) {
    dict_->getLine(in, line, line_values, labels, tags);
    model_->computeHidden(line, line_values, hidden);
    outputProb(std::cout, hidden, labels, tags, threshold);
  }
}

void FastText::getSentenceVector(
    std::istream& in,
    fasttext::Vector& svec) {
  svec.zero();
  std::vector<std::string> tags;
  if (args_->model == model_name::sup) {
    std::vector<int32_t> line, labels;
    std::vector<real> line_values;
    dict_->getLine(in, line, line_values, labels, tags);
    for (int32_t i = 0; i < line.size(); i++) {
      addInputVector(svec, line[i]);
    }
    if (!line.empty()) {
      svec.mul(1.0 / line.size());
    }
  } else {
    Vector vec(args_->dim);
    std::string sentence;
    std::getline(in, sentence);
    std::istringstream iss(sentence);
    std::string word;
    int32_t count = 0;
    while (iss >> word) {
      getWordVector(vec, word);
      real norm = vec.norm();
      if (norm > 0) {
        vec.mul(1.0 / norm);
        svec.addVector(vec);
        count++;
      }
    }
    if (count > 0) {
      svec.mul(1.0 / count);
    }
  }
}

void FastText::ngramVectors(std::string word) {
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  Vector vec(args_->dim);
  dict_->getSubwords(word, ngrams, substrings);
  for (int32_t i = 0; i < ngrams.size(); i++) {
    vec.zero();
    if (ngrams[i] >= 0) {
      if (quant_) {
        vec.addRow(*qinput_, ngrams[i]);
      } else {
        vec.addRow(*input_, ngrams[i]);
      }
    }
    std::cout << substrings[i] << " " << vec << std::endl;
  }
}

void FastText::precomputeWordVectors(Matrix& wordVectors) {
  Vector vec(args_->dim);
  wordVectors.zero();
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      wordVectors.addRow(vec, i, 1.0 / norm);
    }
  }
}

void FastText::findNN(
    const Matrix& wordVectors,
    const Vector& queryVec,
    int32_t k,
    const std::set<std::string>& banSet,
    std::vector<std::pair<real, std::string>>& results) {
  results.clear();
  std::priority_queue<std::pair<real, std::string>> heap;
  real queryNorm = queryVec.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    real dp = wordVectors.dotRow(queryVec, i);
    heap.push(std::make_pair(dp / queryNorm, word));
  }
  int32_t i = 0;
  while (i < k && heap.size() > 0) {
    auto it = banSet.find(heap.top().second);
    if (it == banSet.end()) {
      results.push_back(std::pair<real, std::string>(heap.top().first, heap.top().second));
      i++;
    }
    heap.pop();
  }
}

void FastText::analogies(int32_t k) {
  std::string word;
  Vector buffer(args_->dim), query(args_->dim);
  Matrix wordVectors(dict_->nwords(), args_->dim);
  precomputeWordVectors(wordVectors);
  std::set<std::string> banSet;
  std::cout << "Query triplet (A - B + C)? ";
  std::vector<std::pair<real, std::string>> results;
  while (true) {
    banSet.clear();
    query.zero();
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, -1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);

    findNN(wordVectors, query, k, banSet, results);
    for (auto& pair : results) {
      std::cout << pair.second << " " << pair.first << std::endl;
    }
    std::cout << "Query triplet (A - B + C)? ";
  }
}

void FastText::trainThread(int32_t threadId) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, lossLayer_, threadId);
  if(lossLayer_ == nullptr){
    if (args_->model == model_name::sup) {
      model.setTargetCounts(dict_->getCounts(entry_type::label));
    } else {
      model.setTargetCounts(dict_->getCounts(entry_type::word));
    }
  }

  const int64_t ntokens = dict_->ntokens();
  int64_t localTokenCount = 0;
  real weight = 1;
  std::vector<int32_t> line, labels;
  std::vector<real> line_values;
  std::vector<std::string> tags;

  while (tokenCount_ < args_->epoch * ntokens) {
    real progress = real(tokenCount_) / (args_->epoch * ntokens);
    real lr = args_->lr * (args_->lrDecay != 1.0
            ? std::pow((1.0 - std::pow(progress, args_->lrDecay)), 1.0/args_->lrDecay)
            : 1.0 - progress);
    if (args_->model == model_name::sup) {
      weight = dict_->getLine(ifs, line, line_values, labels, tags);
      localTokenCount += line.size() + labels.size();
      supervised(model, weight * lr, line, line_values, labels);
    } else if (args_->model == model_name::cbow) {
      localTokenCount = dict_->getLine(ifs, line, model.rng);
      //localTokenCount = line.size();
      cbow(model, lr, line);
    } else if (args_->model == model_name::sg) {
      localTokenCount = dict_->getLine(ifs, line, model.rng);
      //localTokenCount = line.size();
      skipgram(model, lr, line);
    }

    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount_ += localTokenCount;
      localTokenCount = 0;
      if (threadId == 0 && args_->verbose > 1)
        loss_ = model.getLoss();
    }
  }
  if (threadId == 0)
    loss_ = model.getLoss();
  ifs.close();
}

void FastText::loadVectors(std::string filename) {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    throw std::invalid_argument(
        "Dimension of pretrained vectors (" + std::to_string(dim) +
        ") does not match dimension (" + std::to_string(args_->dim) + ")!");
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->at(i, j);
    }
  }
  in.close();

  dict_->threshold(1, 0);
  dict_->init();
  input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords()) continue;
    for (size_t j = 0; j < dim; j++) {
      input_->at(idx, j) = mat->at(i, j);
    }
  }
}

void FastText::train(const Args args) {
  args_ = std::make_shared<Args>(args);
  dict_ = std::make_shared<Dictionary>(args_);

  if (args_->verbose > 2) {
    std::cerr << "Training ...\n";
    args_->printInfo();
    std::cerr << "Reading input file ...\n";
  }

  if (args_->input == "-") {
    // manage expectations
    throw std::invalid_argument("Cannot use stdin for training!");
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    throw std::invalid_argument(
        args_->input + " cannot be opened for training!");
  }
  dict_->readFromFile(ifs);
  ifs.close();

  if (args_->pretrainedVectors.size() != 0) {
    loadVectors(args_->pretrainedVectors);
  } else {
    input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
    if(args_->initZeros) input_->zero();
    else input_->uniform(1.0 / args_->dim);
  }

  if (args_->verbose > 2)
    std::cerr << "  Input: " << input_->rows() << " x " << input_->cols()
              << " (" << input_->size() * sizeof(real) / 1024 / 1024 << "M)\n";

  if (args_->verbose > 2)
    std::cerr << "Setting up loss layer ...\n";

  lossLayer_ = lossLayerFactory(args_);
  if(lossLayer_ != nullptr)
    lossLayer_->setup(dict_, args_->seed);

  if (args_->model == model_name::sup) {
    if(lossLayer_ != nullptr)
      output_ = std::make_shared<Matrix>(lossLayer_->getSize(), args_->dim);
    else
      output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
  } else {
    output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
  }
  output_->zero();

  if (args_->verbose > 2)
    std::cerr << "  Output: " << output_->rows() << " x " << output_->cols()
              << " (" << output_->size() * sizeof(real) / 1024 / 1024 << "M)\n";

  startThreads();
  model_ = std::make_shared<Model>(input_, output_, args_, lossLayer_, 0);
  if(lossLayer_ == nullptr){
    if (args_->model == model_name::sup) {
      model_->setTargetCounts(dict_->getCounts(entry_type::label));
    } else {
      model_->setTargetCounts(dict_->getCounts(entry_type::word));
    }
  }

  if(lossLayer_ != nullptr)
    lossLayer_->printInfo();
}

void FastText::startThreads() {
  if (args_->verbose > 2)
    std::cerr << "Starting " << args_->thread << " threads ...\n";

  start_ = clock();
  tokenCount_ = 0;
  loss_ = -1;
  std::vector<std::thread> threads;
  for (int32_t i = 0; i < args_->thread; i++) {
    threads.push_back(std::thread([=]() { trainThread(i); }));
  }
  const int64_t ntokens = dict_->ntokens();
  // Same condition as trainThread
  while (tokenCount_ < args_->epoch * ntokens) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (loss_ >= 0 && args_->verbose > 1) {
      real progress = real(tokenCount_) / (args_->epoch * ntokens);
      std::cerr << "\r";
      printInfo(progress, loss_, std::cerr);
    }
  }
  for (int32_t i = 0; i < args_->thread; i++) {
    threads[i].join();
  }
  if (args_->verbose > 0) {
      std::cerr << "\r";
      printInfo(1.0, loss_, std::cerr);
      std::cerr << std::endl;
  }
}

int FastText::getDimension() const {
    return args_->dim;
}

bool FastText::isQuant() const {
  return quant_;
}

}
