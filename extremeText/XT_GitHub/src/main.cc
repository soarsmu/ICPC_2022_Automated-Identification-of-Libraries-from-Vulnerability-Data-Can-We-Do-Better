/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>
#include <queue>
#include <iomanip>
#include "fasttext.h"
#include "args.h"

using namespace fasttext;

void printUsage() {
  std::cerr
    << "usage: extremetext <command> <args>\n\n"
    << "The commands supported by extremetext are:\n\n"
    << "  supervised              train a supervised classifier\n"
    << "  quantize                quantize a model to reduce the memory usage\n"
    << "  test                    evaluate a supervised classifier\n"
    << "  predict                 predict most likely labels\n"
    << "  predict-prob            predict most likely labels with probabilities\n"
    << "  get-prob                predict probabilities for given labels\n"
    << "  skipgram                train a skipgram model\n"
    << "  cbow                    train a cbow model\n"
    << "  print-word-vectors      print word vectors given a trained model\n"
    << "  print-sentence-vectors  print sentence vectors given a trained model\n"
    << "  print-ngrams            print ngrams given a trained model and word\n"
    << "  nn                      query for nearest neighbors\n"
    << "  analogies               query for analogies\n"
    << "  dump                    dump arguments,dictionary,input/output vectors\n"
//    << "  save-word-vectors       save word vectors given a trained model\n"
//    << "  save-document-vectors   save document vectors given an input file and trained model\n"
    << std::endl;
}

void printQuantizeUsage() {
  std::cerr
    << "usage: extremetext quantize <args>"
    << std::endl;
}

void printTestUsage() {
  std::cerr
    << "usage: extremetext test <model> <test-data> [<k>] [<th>] [<thread>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << "  <th>         (optional; 0.0 by default) probability threshold\n"
    << "  <thread>     (optional; #cpu by default) number of threads\n"
    << std::endl;
}

void printPredictUsage() {
  std::cerr
    << "usage: extremetext predict[-prob] <model> <test-data> [<k>] [<th>] [<output>] [<thread>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << "  <th>         (optional; 0.0 by default) probability threshold\n"
    << "  <output>     (optional; stdout by default) output filename (required by thread options)\n"
    << "  <thread>     (optional; #cpu by default) number of threads\n"
    << std::endl;
}

void printPrintWordVectorsUsage() {
  std::cerr
    << "usage: extremetext print-word-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintSentenceVectorsUsage() {
  std::cerr
    << "usage: extremetext print-sentence-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintNgramsUsage() {
  std::cerr
    << "usage: extremetext print-ngrams <model> <word>\n\n"
    << "  <model>      model filename\n"
    << "  <word>       word to print\n"
    << std::endl;
}

void quantize(const std::vector<std::string>& args) {
  Args a = Args();
  if (args.size() < 3) {
    printQuantizeUsage();
    a.printHelp();
    exit(EXIT_FAILURE);
  }
  a.parseArgs(args);
  FastText fasttext;
  // parseArgs checks if a->output is given.
  fasttext.loadModel(a.output + ".bin");
  fasttext.quantize(a);
  fasttext.saveModel();
  exit(0);
}

void printNNUsage() {
  std::cout
    << "usage: extremetext nn <model> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
}

void printAnalogiesUsage() {
  std::cout
    << "usage: extremetext analogies <model> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
}

void printDumpUsage() {
  std::cout
    << "usage: extremetext dump <model> <option>\n\n"
    << "  <model>      model filename\n"
    << "  <option>     option from args,dict,input,output"
    << std::endl;
}

void printGetProbUsage() {
  std::cout
    << "usage: extremetext get-prob <model> <input> [<th>] [<output>] [<thread>]\n\n"
    << "  <model>      model filename\n"
    << "  <data>       input filename\n"
    << "  <th>         (optional; 0.0 by default) probability threshold\n"
    << "  <output>     (optional; stdout by default) output filename (required by thread options)\n"
    << "  <thread>     (optional; #cpu by default) number of threads\n"
    << std::endl;
}

void printSaveWordVectorsUsage() {
  std::cout
    << "usage: extremetext <model> [<output>]\n\n"
    << "  <model>      model filename\n"
    << "  <output>     (optional; stdout by default) output filename\n"
    << std::endl;
}

void printSaveDocumentVectorsUsage() {
    std::cout
      << "usage: extremetext <model> [<output>]\n\n"
      << "  <model>      model filename\n"
      << "  <data>       input filename with documents\n"
      << "  <output>     (optional; stdout by default) output filename\n"
      << std::endl;
}

void test(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 7) {
    printTestUsage();
    exit(EXIT_FAILURE);
  }

  int32_t k = 1;
  real threshold = 0.0;
  int32_t thread = utils::cpuCount();
  if (args.size() > 4)
    k = std::stoi(args[4]);
  if (args.size() > 5)
    threshold = std::stof(args[5]);
  if (args.size() > 6)
    thread = std::stoi(args[6]);

  FastText fasttext;
  fasttext.loadModel(args[2]);

  std::tuple<int64_t, double, double, double> result;
  std::string infile = args[3];
  if (infile == "-") {
    result = fasttext.test(std::cin, k, threshold);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if(thread == 1) result = fasttext.test(ifs, k, threshold);
    ifs.close();
    if(thread > 1) result = fasttext.testInThreads(infile, k, thread, threshold);
  }
  std::cout << "Number of documents: " << std::get<0>(result) << std::endl;
  std::cout << std::setprecision(5);
  std::cout << "P@" << k << ": " << std::get<1>(result) << std::endl;
  std::cout << "R@" << k << ": " << std::get<2>(result) << std::endl;
  std::cout << "C@" << k << ": " << std::get<3>(result) << std::endl;
}

void predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 8) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }

  int32_t k = 1;
  real threshold = 0.0;
  std::string outfile = "";
  int32_t thread = utils::cpuCount();
  if (args.size() > 4)
    k = std::stoi(args[4]);
  if (args.size() > 5)
    threshold = std::stof(args[5]);
  if (args.size() > 6)
    outfile = std::string(args[6]);
  if (args.size() > 7)
    thread = std::stoi(args[7]);

  bool print_prob = args[1] == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string infile(args[3]);

  if (infile == "-") {
    fasttext.predict(std::cin, k, print_prob, threshold);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if(thread == 1) fasttext.predict(ifs, k, print_prob, threshold);
    ifs.close();
    if(thread > 1) fasttext.predictInThreads(infile, outfile, thread, k, print_prob, threshold);
  }

  exit(0);
}

void getProb(const std::vector<std::string>& args){
  if (args.size() < 4 || args.size() > 7) {
    printGetProbUsage();
    exit(EXIT_FAILURE);
  }

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string infile(args[3]);
  std::string outfile = "";
  int32_t thread = utils::cpuCount();
  real threshold = 0.0;
  if(args.size() > 4)
    threshold = std::stof(args[4]);
  if(args.size() > 5)
    outfile = std::string(args[5]);
  if(args.size() > 6)
    thread = std::stoi(args[6]);


  if (infile == "-") {
    fasttext.getProb(std::cin, threshold);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if(thread == 1) fasttext.getProb(ifs, threshold);
    ifs.close();
    if(thread > 1) fasttext.getProbInThreads(infile, outfile, thread, threshold);
  }

  exit(0);
}

void printWordVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintWordVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string word;
  Vector vec(fasttext.getDimension());
  while (std::cin >> word) {
    fasttext.getWordVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
  exit(0);
}

void printSentenceVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintSentenceVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  Vector svec(fasttext.getDimension());
  while (std::cin.peek() != EOF) {
    fasttext.getSentenceVector(std::cin, svec);
    // Don't print sentence
    std::cout << svec << std::endl;
  }
  exit(0);
}

void printNgrams(const std::vector<std::string> args) {
  if (args.size() != 4) {
    printPrintNgramsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.ngramVectors(std::string(args[3]));
  exit(0);
}

void nn(const std::vector<std::string> args) {
  int32_t k;
  if (args.size() == 3) {
    k = 10;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
  } else {
    printNNUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string queryWord;
  std::shared_ptr<const Dictionary> dict = fasttext.getDictionary();
  Vector queryVec(fasttext.getDimension());
  Matrix wordVectors(dict->nwords(), fasttext.getDimension());
  std::cerr << "Pre-computing word vectors...";
  fasttext.precomputeWordVectors(wordVectors);
  std::cerr << " done." << std::endl;
  std::set<std::string> banSet;
  std::cout << "Query word? ";
  std::vector<std::pair<real, std::string>> results;
  while (std::cin >> queryWord) {
    banSet.clear();
    banSet.insert(queryWord);
    fasttext.getWordVector(queryVec, queryWord);
    fasttext.findNN(wordVectors, queryVec, k, banSet, results);
    for (auto& pair : results) {
      std::cout << pair.second << " " << pair.first << std::endl;
    }
    std::cout << "Query word? ";
  }
  exit(0);
}

void analogies(const std::vector<std::string> args) {
  int32_t k;
  if (args.size() == 3) {
    k = 10;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
  } else {
    printAnalogiesUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.analogies(k);
  exit(0);
}

void train(const std::vector<std::string> args) {
  Args a = Args();
  a.train = true;
  a.parseArgs(args);
  FastText fasttext;
  std::ofstream ofs(a.output + ".bin");
  if (!ofs.is_open()) {
    throw std::invalid_argument(a.output + ".bin cannot be opened for saving.");
  }
  ofs.close();
  fasttext.train(a);
  fasttext.saveModel();
  if (a.saveVectors)
    fasttext.saveVectors();
    //fasttext.saveVectors(a.output + ".vec");
  if (a.saveOutput)
    fasttext.saveOutput();
  //if (a.saveDocuments)
    //fasttext.saveDocuments(a.input, a.output + ".vec");
}

void dump(const std::vector<std::string>& args) {
  if (args.size() < 4) {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }

  std::string modelPath = args[2];
  std::string option = args[3];

  FastText fasttext;
  fasttext.loadModel(modelPath);
  if (option == "args") {
    fasttext.getArgs().dump(std::cout);
  } else if (option == "dict") {
    fasttext.getDictionary()->dump(std::cout);
  } else if (option == "input") {
    if (fasttext.isQuant()) {
      std::cerr << "Not supported for quantized models." << std::endl;
    } else {
      fasttext.getInputMatrix()->dump(std::cout);
    }
  } else if (option == "output") {
    if (fasttext.isQuant()) {
      std::cerr << "Not supported for quantized models." << std::endl;
    } else {
      fasttext.getOutputMatrix()->dump(std::cout);
    }
  } else {
    printDumpUsage();
    exit(EXIT_FAILURE);
  }
}

// TODO
void saveWordVectors(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        printSaveWordVectorsUsage();
        exit(EXIT_FAILURE);
    }

    FastText fasttext;
    fasttext.loadModel(args[2]);
    //fasttext.saveVectors(args[3]);
}

void saveDocumentVectors(const std::vector<std::string>& args) {
    if (args.size() < 3) {
        printSaveDocumentVectorsUsage();
        exit(EXIT_FAILURE);
    }

    FastText fasttext;
    fasttext.loadModel(args[2]);
    //fasttext.saveDocuments(args[3], args[4]);
}

int main(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(args[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised") {
    train(args);
  } else if (command == "test") {
    test(args);
  } else if (command == "quantize") {
    quantize(args);
  } else if (command == "print-word-vectors") {
    printWordVectors(args);
  } else if (command == "print-sentence-vectors") {
    printSentenceVectors(args);
  } else if (command == "print-ngrams") {
    printNgrams(args);
  } else if (command == "nn") {
    nn(args);
  } else if (command == "analogies") {
    analogies(args);
  } else if (command == "predict" || command == "predict-prob") {
    predict(args);
  } else if (command == "get-prob") {
    getProb(args);
  } else if (command == "dump") {
    dump(args);
//  } else if (command == "save-word-vectors") {
//    saveWordVectors(args);
//  } else if (command == "save-document-vectors") {
//    saveDocumentVectors(args);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
