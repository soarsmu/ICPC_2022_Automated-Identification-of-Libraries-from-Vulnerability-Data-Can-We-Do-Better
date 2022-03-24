/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "utils.h"

#include <ios>
#include <thread>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iostream>

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }

  size_t cpuCount(){
    return std::thread::hardware_concurrency();
  }

  std::string itos(int32_t number, int32_t leadingZeros){
      std::stringstream ss;
      if (leadingZeros != 0) ss << std::setw(leadingZeros) << std::setfill('0');
      ss << number;
      return ss.str();
  }

  void printProgress(int64_t start, int64_t value, int64_t end, std::ostream& log_stream){
    int64_t progress = value - start;
    int64_t range = end - start;
    float progressProc = static_cast<float>(progress) / range * 100;
    if(progress % std::max((int64_t)1, range / 1000) == 0){
      log_stream << std::fixed;
      log_stream << "Progress: ";
      log_stream << std::setprecision(1) << std::setw(5) << progressProc << "%\r";
      log_stream << std::flush;
    }
  }

  void printProgress(float progress, std::ostream& log_stream) {
    progress = progress * 100;
    log_stream << std::fixed;
    log_stream << "Progress: ";
    log_stream << std::setprecision(1) << std::setw(5) << progress << "%\r";
    log_stream << std::flush;
  }

  uint32_t hash(const char* data, size_t size){
      uint32_t h = 2166136261;
      for (size_t i = 0; i < size; i++) {
          h = h ^ uint32_t(data[i]);
          h = h * 16777619;
      }
      return h;
  }

  void loadString(std::istream& in, std::string& str){
      size_t str_size;
      in.read((char*) &(str_size), sizeof(str_size));
      str.resize(str_size);
      in.read(&(str[0]), str_size * sizeof(char));
  }

  void saveString(std::ostream& out, std::string& str){
      size_t str_size = str.length();
      out.write((char*) &(str_size), sizeof(str_size));
      out.write(&(str[0]), str_size * sizeof(char));
  }
}

}
