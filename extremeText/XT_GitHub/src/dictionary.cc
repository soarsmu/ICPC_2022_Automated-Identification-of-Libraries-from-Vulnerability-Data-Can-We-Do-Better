/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "dictionary.h"

#include <assert.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <stdexcept>

#include "utils.h"
#include "sutils.h"

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args) : args_(args),
  word2int_(MAX_VOCAB_SIZE, -1), size_(0), nwords_(0), nlabels_(0),
  ntokens_(0), ndocs_(0), pruneidx_size_(-1) {}

Dictionary::Dictionary(std::shared_ptr<Args> args, std::istream& in) : args_(args),
  size_(0), nwords_(0), nlabels_(0), ntokens_(0), ndocs_(0), pruneidx_size_(-1) {
  load(in);
}

int32_t Dictionary::find(const std::string& w) const {
  return find(w, utils::hash(w));
}

int32_t Dictionary::find(const std::string& w, uint32_t h) const {
  int32_t word2intsize = word2int_.size();
  int32_t id = h % word2intsize;
  while (word2int_[id] != -1 && words_[word2int_[id]].word != w) {
    id = (id + 1) % word2intsize;
  }
  return id;
}

int32_t Dictionary::add(const std::string& w, int32_t c) {
  int32_t h = find(w);
  ntokens_ += c;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = c;
    e.doc_count = 0;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++;
    if (e.type == entry_type::word) nwords_++;
    else if (e.type == entry_type::label) nlabels_++;
  } else {
    words_[word2int_[h]].count += c;
  }

  return h;
}

void Dictionary::add(const std::string& w, int32_t c, std::unordered_set<int32_t>& uniqWords) {
  int32_t h = add(w, c);
  if(!uniqWords.count(h)){
    words_[word2int_[h]].doc_count++;
    uniqWords.insert(h);
  }
}

int32_t Dictionary::nwords() const {
  return nwords_;
}

int32_t Dictionary::nlabels() const {
  return nlabels_;
}

int64_t Dictionary::ntokens() const {
  return ntokens_;
}

int32_t Dictionary::ndocs() const {
  return ndocs_;
}

const std::vector<int32_t>& Dictionary::getSubwords(int32_t i) const {
  assert(i >= 0);
  assert(i < nwords_);
  return words_[i].subwords;
}

const std::vector<int32_t> Dictionary::getSubwords(
    const std::string& word) const {
  int32_t i = getId(word);
  if (i >= 0) {
    return getSubwords(i);
  }
  std::vector<int32_t> ngrams;
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams);
  }
  return ngrams;
}

void Dictionary::getSubwords(const std::string& word,
                           std::vector<int32_t>& ngrams,
                           std::vector<std::string>& substrings) const {
  int32_t i = getId(word);
  ngrams.clear();
  substrings.clear();
  if (i >= 0) {
    ngrams.push_back(i);
    substrings.push_back(words_[i].word);
  }
  if (word != EOS) {
    computeSubwords(BOW + word + EOW, ngrams, substrings);
  }
}

bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) return false;
  return rand > pdiscard_[id];
}

int32_t Dictionary::getId(const std::string& w, uint32_t h) const {
  int32_t id = find(w, h);
  return word2int_[id];
}

int32_t Dictionary::getId(const std::string& w) const {
  int32_t h = find(w);
  return word2int_[h];
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].type;
}

entry_type Dictionary::getType(const std::string& w) const {
  return w.find(args_->label) == 0 ? entry_type::label : entry_type::word;
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return words_[id].word;
}

void Dictionary::computeSubwords(const std::string& word,
                               std::vector<int32_t>& ngrams,
                               std::vector<std::string>& substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = utils::hash(ngram) % args_->bucket;
        ngrams.push_back(nwords_ + h);
        substrings.push_back(ngram);
      }
    }
  }
}

void Dictionary::computeSubwords(const std::string& word,
                               std::vector<int32_t>& ngrams) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = utils::hash(ngram) % args_->bucket;
        pushHash(ngrams, h);
      }
    }
  }
}

void Dictionary::initNgrams() {
  for (int32_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);
    }
  }
}

bool Dictionary::readWord(std::istream& in, std::string& word) const
{
  real value;
  return readWord(in, word, value);
}

bool Dictionary::readWord(std::istream& in, std::string& word, real& value) const
{
  bool val_part = false;
  std::string val;
  value = 1;

  int c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          value = args_->eosWeight;
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        if(args_->wordsWeights && !val.empty())
          value = std::stof(val);
        return true;
      }
    }
    if (val_part) val.push_back(c);
    else if (c == ':') val_part = true;
    else word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  real value;
  int64_t minThreshold = 1;
  std::unordered_set<int32_t> uniqWords; // For TF-IDF
  while (readWord(in, word, value)) {
    if(word == args_->weight || word.find(args_->tag) == 0){
      continue;
    }
    if(args_->tfidfWeights) add(word, static_cast<int32_t>(value), uniqWords);
    else add(word, static_cast<int32_t>(value));
    if(word == EOS) {
      uniqWords.clear();
      ++ndocs_;
    }
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
    }
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold, minThreshold);
    }
  }
  ++ndocs_;
  threshold(args_->minCount, args_->minCountLabel);
  initTableDiscard();
  initNgrams();
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
    std::cerr << "Number of documents: " << ndocs_ << std::endl;
    std::cerr << "Number of words: " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}

void Dictionary::threshold(int64_t t, int64_t tl) {
  //std::cerr << "Removing words and labels under min count threshold ...\n";
  sort(words_.begin(), words_.end(), [](const entry& e1, const entry& e2) {
      if (e1.type != e2.type) return e1.type < e2.type;
      return e1.count > e2.count;
    });
  words_.erase(remove_if(words_.begin(), words_.end(), [&](const entry& e) {
        return (e.type == entry_type::word && e.count < t) ||
               (e.type == entry_type::label && e.count < tl);
      }), words_.end());
  words_.shrink_to_fit();
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(word2int_.begin(), word2int_.end(), -1);
  for (auto it = words_.begin(); it != words_.end(); ++it) {
    int32_t h = find(it->word);
    word2int_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (int32_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}

std::vector<int64_t> Dictionary::getCounts(entry_type type) const {
  std::vector<int64_t> counts;
  for (auto& w : words_) {
    if (w.type == type) counts.push_back(w.count);
  }
  return counts;
}

void Dictionary::addWordNgrams(std::vector<int32_t>& line,
                               const std::vector<int32_t>& hashes,
                               int32_t n) const {
  for (size_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (size_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      pushHash(line, h % args_->bucket);
    }
  }
}

void Dictionary::addSubwords(std::vector<int32_t>& line,
                             const std::string& token,
                             std::vector<real>& line_values,
                             const real value,
                             std::vector<int32_t>& doc_counts,
                             int32_t wid) const {
  if (wid < 0) { // out of vocab
    if (token != EOS) {
      computeSubwords(BOW + token + EOW, line);
    }
  } else {
    if (args_->maxn <= 0) { // in vocab w/o subwords
      line.push_back(wid);
      line_values.push_back(value);
      doc_counts.push_back(words_[wid].doc_count);
    } else { // in vocab w/ subwords
      const std::vector<int32_t>& ngrams = getSubwords(wid);
      line.insert(line.end(), ngrams.cbegin(), ngrams.cend());
      for(auto it = 0; it < ngrams.size(); ++it) {
        line_values.push_back(value);
        doc_counts.push_back(words_[wid].doc_count);
      }
    }
  }
}

void Dictionary::reset(std::istream& in) const {
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }
}

int32_t Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  real value;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  while (readWord(in, token, value)) {
    int32_t h = find(token);
    int32_t wid = word2int_[h];
    if (wid < 0) continue;

    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) break;
  }
  return ntokens;
}

real Dictionary::getLine(std::istream& in,
                            std::vector<int32_t>& words,
                            std::vector<real>& words_values,
                            std::vector<int32_t>& labels,
                            std::vector<std::string>& tags) const {
  std::vector<int32_t> word_hashes;
  std::string token;
  real weight = 1.0;
  real value;
  int32_t ntokens = 0;

  reset(in);
  words.clear();
  labels.clear();
  words_values.clear();
  tags.clear(); // Document tags
  std::vector<int32_t> doc_counts; // For TF-IDF

  while (readWord(in, token, value)) {
    if(token == args_->weight){
      weight = value;
      continue;
    }
    if(token.find(args_->tag) == 0) {
      tags.push_back(token);
      continue;
    }
    if(token == EOS && !args_->addEosToken) break;

    uint32_t h = utils::hash(token);
    int32_t wid = getId(token, h);
    entry_type type = wid < 0 ? getType(token) : getType(wid);

    ntokens++;
    if (type == entry_type::word) {
      addSubwords(words, token, words_values, value, doc_counts, wid);
      word_hashes.push_back(h);
    } else if (type == entry_type::label && wid >= 0) {
      labels.push_back(wid - nwords_);
    }

    if (token == EOS) break;
  }

  if(args_->tfidfWeights){
    assert(words.size() == doc_counts.size());
    int32_t nwords = 0;
    std::unordered_map<int32_t, std::pair<int32_t, int32_t>> counts;
    for(size_t i = 0; i < words.size(); ++i){
      auto word_count = counts.find(words[i]);
      if(word_count == counts.end()) counts[words[i]] = std::pair<int32_t, int32_t>(static_cast<int32_t>(words_values[i]), doc_counts[i]);
      else word_count->second.first += words_values[i];
      nwords += words_values[i];
    }

    words.clear();
    words_values.clear();
    for(auto it : counts){
      words.push_back(it.first);
      real tfidf = (static_cast<real>(it.second.first) / nwords) * std::log(static_cast<real>(ndocs_) / (1 + it.second.second));
      words_values.push_back(tfidf);
    }
  }

  // TODO: support for wordNgrams for TF-IDF and wordsWeights
  if(!args_->tfidfWeights && !args_->wordsWeights) {
    addWordNgrams(words, word_hashes, args_->wordNgrams);
    while (words.size() != words_values.size())
      words_values.push_back(1.0);
  }

  // Filter out weights below weightsThr;
  if(args_->weightsThr > 0){
    size_t r = 0, w = 0, s = words_values.size();
    while(r < s) {
      if (words_values[r] >= args_->weightsThr) {
        words[w] = words[r];
        words_values[w] = words_values[r];
        ++w;
      }
      ++r;
    }
    words.resize(w);
    words_values.resize(w);
  }

  assert(words.size() == words_values.size());

  return weight;
}

void Dictionary::pushHash(std::vector<int32_t>& hashes, int32_t id) const {
  if (pruneidx_size_ == 0 || id < 0) return;
  if (pruneidx_size_ > 0) {
    if (pruneidx_.count(id)) {
      id = pruneidx_.at(id);
    } else {
      return;
    }
  }
  hashes.push_back(nwords_ + id);
}

std::string Dictionary::getLabel(int32_t lid) const {
  if (lid < 0 || lid >= nlabels_) {
    throw std::invalid_argument(
        "Label id is out of range [0, " + std::to_string(nlabels_) + "]");
  }
  return words_[lid + nwords_].word;
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*) &size_, sizeof(int32_t));
  out.write((char*) &nwords_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &ntokens_, sizeof(int64_t));
  out.write((char*) &ndocs_, sizeof(int32_t));
  out.write((char*) &pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = words_[i];
    out.write(e.word.data(), e.word.size() * sizeof(char));
    out.put(0);
    out.write((char*) &(e.count), sizeof(int64_t));
    out.write((char*) &(e.doc_count), sizeof(int32_t));
    out.write((char*) &(e.type), sizeof(entry_type));
  }
  for (const auto pair : pruneidx_) {
    out.write((char*) &(pair.first), sizeof(int32_t));
    out.write((char*) &(pair.second), sizeof(int32_t));
  }
}

void Dictionary::load(std::istream& in) {
  words_.clear();
  in.read((char*) &size_, sizeof(int32_t));
  in.read((char*) &nwords_, sizeof(int32_t));
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &ntokens_, sizeof(int64_t));
  in.read((char*) &ndocs_, sizeof(int32_t));
  in.read((char*) &pruneidx_size_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.word.push_back(c);
    }
    in.read((char*) &e.count, sizeof(int64_t));
    in.read((char*) &e.doc_count, sizeof(int32_t));
    in.read((char*) &e.type, sizeof(entry_type));
    words_.push_back(e);
  }
  pruneidx_.clear();
  for (int32_t i = 0; i < pruneidx_size_; i++) {
    int32_t first;
    int32_t second;
    in.read((char*) &first, sizeof(int32_t));
    in.read((char*) &second, sizeof(int32_t));
    pruneidx_[first] = second;
  }
  initTableDiscard();
  initNgrams();

  int32_t word2intsize = std::ceil(size_ / 0.7);
  word2int_.assign(word2intsize, -1);
  for (int32_t i = 0; i < size_; i++) {
    word2int_[find(words_[i].word)] = i;
  }
}

void Dictionary::init() {
  initTableDiscard();
  initNgrams();
}

void Dictionary::prune(std::vector<int32_t>& idx) {
  std::vector<int32_t> words, ngrams;
  for (auto it = idx.cbegin(); it != idx.cend(); ++it) {
    if (*it < nwords_) {words.push_back(*it);}
    else {ngrams.push_back(*it);}
  }
  std::sort(words.begin(), words.end());
  idx = words;

  if (ngrams.size() != 0) {
    int32_t j = 0;
    for (const auto ngram : ngrams) {
      pruneidx_[ngram - nwords_] = j;
      j++;
    }
    idx.insert(idx.end(), ngrams.begin(), ngrams.end());
  }
  pruneidx_size_ = pruneidx_.size();

  std::fill(word2int_.begin(), word2int_.end(), -1);

  int32_t j = 0;
  for (size_t i = 0; i < words_.size(); i++) {
    if (getType(i) == entry_type::label || (j < words.size() && words[j] == i)) {
      words_[j] = words_[i];
      word2int_[find(words_[j].word)] = j;
      j++;
    }
  }
  nwords_ = words.size();
  size_ = nwords_ +  nlabels_;
  words_.erase(words_.begin() + size_, words_.end());
  initNgrams();
}

void Dictionary::dump(std::ostream& out) const {
  out << words_.size() << std::endl;
  for (auto it : words_) {
    std::string entryType = "word";
    if (it.type == entry_type::label) {
      entryType = "label";
    }
    out << it.word << " " << it.count << " " << entryType << std::endl;
  }
}

std::vector<std::string> Dictionary::getWords(entry_type type) const {
  std::vector<std::string> words;
  for (auto& w : words_) {
    if (w.type == type) words.push_back(w.word);
  }
  return words;
}

}
