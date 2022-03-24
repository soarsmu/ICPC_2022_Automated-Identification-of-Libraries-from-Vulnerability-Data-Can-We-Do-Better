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

#include <fstream>
#include <vector>
#include <string>

#if defined(__clang__) || defined(__GNUC__)
# define FASTTEXT_DEPRECATED(msg) __attribute__((__deprecated__(msg)))
#elif defined(_MSC_VER)
# define FASTTEXT_DEPRECATED(msg) __declspec(deprecated(msg))
#else
# define FASTTEXT_DEPRECATED(msg)
#endif

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream&);
  void seek(std::ifstream&, int64_t);
  size_t cpuCount();
  std::string itos(int32_t, int32_t = 0);
  void printProgress(int64_t start, int64_t value, int64_t end, std::ostream& log_stream);
  void printProgress(float progress, std::ostream& log_stream);

  uint32_t hash(const char*, size_t size);
  inline uint32_t hash(const std::string& str){ return hash(str.data(), str.length()); };

  void loadString(std::istream& in, std::string& str);
  void saveString(std::ostream& out, std::string& str);

  template <typename T>
  void printVector(std::vector<T> vec, std::ostream& out_stream){
    for(size_t i = 0; i < vec.size(); ++i) {
        if(i != 0) out_stream << ", ";
        out_stream << vec[i];
    }
  }
}

}
