/**
 * Copyright (c) 2018 by Marek Wydmuch, Róbert Busa-Fekete, Krzysztof Dembczyński
 * All rights reserved.
 */
 
#include <fstream>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <list>
#include <chrono>
#include <random>
#include <climits>
#include <iomanip>
#include <array>

#include "loss_plt.h"
#include "model.h"
#include "threads.h"
#include "utils.h"

namespace fasttext {

PLT::PLT(std::shared_ptr<Args> args) : LossLayer(args){
    multilabel = true;

    // Stats
    n_in_vis_count = 0;
    n_vis_count = 0;
    y_count = 0;
    x_count = 0;
}

PLT::~PLT() {
    for(size_t i = 0; i < tree.size(); ++i){
        delete tree[i];
    }
}

void PLT::buildHuffmanPLTree(const std::vector<int64_t>& freq){
    if(args_->verbose > 2)
        std::cerr << "  Building PLT with Huffman tree ...\n";

    k = freq.size();
    t = 2 * k - 1; // size of the tree

    std::priority_queue<NodeFreq, std::vector<NodeFreq>, std::greater<NodeFreq>> freq_heap;
    for(int i = 0; i < k; ++i) {
        NodePLT *n = createNode(nullptr, i);
        freq_heap.push({n, freq[i]});
        
        //std::cerr << "Leaf: " << n->label << ", Node: " << n->n << ", Freq: " << freq[i] << "\n";
    }

    while (true) {
        std::vector<NodeFreq> toMerge;
        for (int a = 0; a < args_->arity; ++a) {
            NodeFreq tmp = freq_heap.top();
            freq_heap.pop();
            toMerge.push_back(tmp);
            if (freq_heap.empty()) break;
        }

        NodePLT *parent = createNode();

        int64_t aggregatedFrequency = 0;
        for (NodeFreq e : toMerge) {
            e.node->parent = parent;
            parent->children.push_back(e.node);
            aggregatedFrequency += e.freq;
        }

        if (freq_heap.empty()) {
            tree_root = parent;
            tree_root->parent = nullptr;
            break;
        }

        freq_heap.push({parent, aggregatedFrequency});
    }

    t = tree.size();
    std::cerr << "    Nodes: " << tree.size() << ", leaves: " << tree_labels.size() << ", arity: " << args_->arity << "\n";
}

void PLT::buildCompletePLTree(int32_t k_) {
  if(args_->verbose > 2)
    std::cerr << "  Building PLT with complete tree ...\n";

  // Build complete tree
  k = k_;
  t = static_cast<int>(ceil(static_cast<double>(args_->arity * k - 1) / (args_->arity - 1)));
  uint32_t ti = t - k;

  std::vector<int32_t> labels_order;
  if (args_->randomTree){
    for (auto i = 0; i < k; ++i)
      labels_order.push_back(i);
    std::shuffle(labels_order.begin(), labels_order.end(), rng);
  }

  for(size_t i = 0; i < t; ++i){
    NodePLT *n = createNode();

    if(i >= ti){
      if(args_->randomTree) n->label = labels_order[i - ti];
      else n->label = i - ti;
      tree_labels.insert(std::make_pair(n->label, n));
    }

    if(i > 0){
      n->parent = tree[static_cast<int>(floor(static_cast<float>(n->index - 1) / args_->arity))];
      n->parent->children.push_back(n);
    }
  }

  tree_root = tree[0];
  tree_root->parent = nullptr;

  std::cerr << "    Nodes: " << tree.size() << ", leaves: " << tree_labels.size() << ", arity: " << args_->arity << "\n";
}


#define MUTEXES 2048
void featureMatrixThread(int threadId, std::shared_ptr<Dictionary> dict, std::shared_ptr<Args> args,
        std::vector<std::unordered_map<int32_t, real>>& tmpLabelsFeatures, std::array<std::mutex, MUTEXES>& mutexes){

    std::ifstream ifs(args->input);
    std::vector<int32_t> line, labels;
    std::vector<real> line_values;
    std::vector<std::string> tags;

    int64_t insize = utils::size(ifs);
    utils::seek(ifs, threadId * insize / args->thread);
    int64_t startpos = ifs.tellg();
    int64_t endpos = (threadId + 1) * insize / args->thread;

    while (ifs.peek() != EOF) {
        dict->getLine(ifs, line, line_values, labels, tags);
        auto pos = ifs.tellg();
        if(threadId == 0 && args->verbose > 0) utils::printProgress(startpos, pos, endpos, std::cerr);
        if(pos < startpos || pos > endpos) break;

        if(args->kMeansSample < 1.0){
            uint32_t h = utils::hash((char*)(&pos), sizeof(pos));
            uint32_t max = 1 << 24;
            if (args->kMeansSample < static_cast<real>(h % max) / max) continue;
        }

        unitNorm(line_values.data(), line_values.size() - (args->addEosToken ? 1 : 0));
        for(const auto& l : labels){
            std::mutex &m = mutexes[l % mutexes.size()];
            m.lock();
            for(int j = 0; j < line.size(); ++j){
                auto f = tmpLabelsFeatures[l].find(line[j]);
                if(f == tmpLabelsFeatures[l].end())
                    tmpLabelsFeatures[l][line[j]] = line_values[j];
                else (*f).second += line_values[j];
            }
            m.unlock();
        }
    }
    ifs.close();
}
  
NodePartition nodeKMeansThread(NodePartition nPart, SRMatrix<Feature>& labelsFeatures, std::shared_ptr<Args> args, int seed){
  kMeans(nPart.partition, labelsFeatures, args->arity, args->kMeansEps, args->kMeansBalanced, seed);
  return nPart;
}
  
void PLT::buildKMeansPLTree(std::shared_ptr<Args> args, std::shared_ptr<Dictionary> dict){

  // Build label's feature matrix
  tree_root = createNode();
  k = dict->nlabels();

  SRMatrix<Feature> labelsFeatures;

  {
    std::cerr << "Computing labels' features matrix in " << args->thread << " threads ...\n";
    std::vector<std::unordered_map<int32_t, real>> tmpLabelsFeatures(k);

    if(args->thread > 1){
      std::array<std::mutex, MUTEXES> mutexes;
      ThreadSet tSet;
      for(int t = 0; t < args_->thread; ++t)
        tSet.add(featureMatrixThread, t, dict, args, std::ref(tmpLabelsFeatures), std::ref(mutexes));
      tSet.joinAll();
    } else {
      std::ifstream ifs(args->input);
      std::vector<int32_t> line, labels;
      std::vector<real> line_values;
      std::vector<std::string> tags;

      int i = 0;
      while (ifs.peek() != EOF) {
        utils::printProgress(0, i++, dict->ndocs(), std::cerr);
        dict->getLine(ifs, line, line_values, labels, tags);
        unitNorm(line_values.data(), line_values.size() - (args->addEosToken ? 1 : 0));
        for(const auto& l : labels){
          for(int j = 0; j < line.size(); ++j){
            auto f = tmpLabelsFeatures[l].find(line[j]);
            if(f == tmpLabelsFeatures[l].end())
              tmpLabelsFeatures[l][line[j]] = line_values[j];
            else (*f).second += line_values[j];
          }
        }
      }
      ifs.close();
    }

    uint64_t featureCount = 0;
    for(int l = 0; l < k; ++l){
      utils::printProgress(0, l, k, std::cerr);
      std::vector<Feature> labelFeatures;
      for(const auto& f : tmpLabelsFeatures[l])
        labelFeatures.push_back({f.first, f.second});
      std::sort(labelFeatures.begin(), labelFeatures.end());
      unitNorm(labelFeatures);
      featureCount += labelFeatures.size();
      labelsFeatures.appendRow(labelFeatures);
    }

    std::cerr << "  Features matrix: " << featureCount << " ("
              << featureCount * sizeof(int) * sizeof(real) / 1024 / 1024 << "M)\n";

    //assert(labelsFeatures.rows() == dict->nlabels());
    //assert(labelsFeatures.cols() == dict->nwords());
  }

  // Prepare partitions
  std::uniform_int_distribution<int> kMeansSeeder(0, INT_MAX);
  auto partition = new std::vector<Assignation>(k);
  for(int i = 0; i < k; ++i) (*partition)[i].index = i;

  // Run clustering in parallel
  ThreadPool tPool(args->thread);
  std::vector<std::future<NodePartition>> results;

  NodePartition rootPart = {tree_root, partition};
  results.emplace_back(tPool.enqueue(nodeKMeansThread, rootPart, std::ref(labelsFeatures), args, kMeansSeeder(rng)));

  std::cerr << "Starting hierarchical K-Means clustering in " << args->thread << " threads ...\n";

  for(int r = 0; r < results.size(); ++r) {
    utils::printProgress(0, r, results.size(), std::cerr);

    // Enqueuing new clustering tasks in the main thread ensures determinism
    NodePartition nPart = results[r].get();

    // This needs to be done this way in case of imbalanced K-Means
    auto partitions = new std::vector<Assignation>* [args->arity];
    for (int i = 0; i < args->arity; ++i) partitions[i] = new std::vector<Assignation>();
    for (auto a : *nPart.partition) partitions[a.value]->push_back({a.index, 0});

    for (int i = 0; i < args->arity; ++i) {
      if(partitions[i]->empty()) continue;
      else if(partitions[i]->size() == 1){
        createNode(nPart.node, partitions[i]->front().index);
        delete partitions[i];
        continue;
      }

      NodePLT *n = createNode(nPart.node);

      if(partitions[i]->size() <= args->maxLeaves) {
        for (const auto& a : *partitions[i]) createNode(n, a.index);
        delete partitions[i];
      } else {
        NodePartition childPart = {n, partitions[i]};
        results.emplace_back(tPool.enqueue(nodeKMeansThread, childPart, std::ref(labelsFeatures), args, kMeansSeeder(rng)));
      }
    }

    delete nPart.partition;
  }

  t = tree.size();
  assert(k == tree_labels.size());
  std::cerr << "  Nodes: " << tree.size() << ", leaves: " << tree_labels.size() << "\n";
}

void PLT::loadTreeStructure(std::string filename, std::shared_ptr<Dictionary> dict){
  if(args_->verbose > 2)
    std::cerr << "  Loading PLT structure from file: " << filename << " ...\n";
  std::ifstream treefile(filename);

  createNode();
  tree_root = tree[0]; // Node with id 0 is assumed to be a default root node

  std::unordered_map<std::string, int32_t> nodesMap;
  nodesMap.insert({"0", 0});

  std::string line;
  while(std::getline(treefile, line)){
    if(!line.length()) continue;

    int32_t parentId, childId;
    std::string parent, child, label;

    std::istringstream lineISS(line);
    lineISS >> parent >> child >> label;

    if(parent == child)
      throw std::invalid_argument("Node " + child + " can't have itself as a parent!");

    auto c = nodesMap.find(child);
    if(c != nodesMap.end()) childId = c->second;
    else{
      createNode();
      childId = nodesMap.insert(std::make_pair(child, nodesMap.size())).first->second;
    }
    NodePLT *childN = tree[childId];
    if(childN->parent)
      throw std::invalid_argument("Node " + child + " occurs more than once in the tree structure file!");

    // Assign label to the node if present
    if(label.length()){
      dict->add(label, 0);
      int32_t wid = dict->getId(label);

      if(wid >= 0 && dict->getType(wid) == entry_type::label){
        int32_t labelId = wid - dict->nwords();
        childN->label = labelId;
        if(tree_labels.count(labelId))
          throw std::invalid_argument(label + " occurs more than once in the tree structure file!");
        else tree_labels.insert(std::make_pair(childN->label, childN));
      }
    }

    // Set the new root if no parent
    if(parent == "-1"){
      tree_root = tree[childId];
      tree_root->parent = nullptr;
      continue;
    }

    // Update parent
    auto p = nodesMap.find(parent);
    if(p != nodesMap.end()) parentId = p->second;
    else{
      createNode();
      parentId = nodesMap.insert(std::make_pair(parent, nodesMap.size())).first->second;
    }

    NodePLT *parentN = tree[parentId];
    parentN->children.push_back(childN);
    childN->parent = parentN;
  }
  treefile.close();

  // Validate tree structure
  if(args_->verbose > 2)
    std::cerr << "    Validating tree structure ...\n";

  // Searching for cycles
  for(const auto& l : tree_labels){
    NodePLT* n = l.second;
    std::unordered_set<NodePLT*> visited;
    while(n->parent){
      visited.insert(n);
      n = n->parent;

      if(visited.count(n))
        throw std::invalid_argument("There is a cycle in the tree structure!");
    }
    if(n != tree_root)
      throw std::invalid_argument("There is no path to the tree root from " + dict->getWord(l.first + dict->nwords()) + "!");
  }

  // Checking if all labels are present in the tree structure
  for(const auto& l : dict->getWords(entry_type::label)){
    int32_t label_id = dict->getId(l) - dict->nwords();
    if(!tree_labels.count(label_id))
      throw std::invalid_argument(l + " doesn't occur in the tree structure file!");
  }

  if(args_->verbose > 2)
    std::cerr << "    Nodes: " << tree.size() << ", labels: " << tree_labels.size() << "\n";
  t = tree.size();
  k = tree_labels.size();
}

real PLT::learnNode(NodePLT *n, real label, real lr, real l2, Model *model_){
    if(n->label < 0) ++n_in_vis_count;
    ++n_vis_count;
    ++n->n_updates;
    if (label) ++n->n_positive_updates;
    return binaryLogistic(n->index, label, lr, l2, model_);
}

real PLT::predictNode(NodePLT *n, Vector& hidden, const Model *model_){
    if(n->n_updates == 0 || n->n_positive_updates == 0) return 0;
    else if(n->n_positive_updates == n->n_updates) return 1;
    else return model_->sigmoid(model_->wo_->dotRow(hidden, shift + n->index));
}

NodePLT* PLT::createNode(NodePLT *parent, int32_t label){
    NodePLT *n = new NodePLT();
    n->index = tree.size();
    n->label = label;
    n->parent = parent;
    n->n_updates = 0;
    n->n_positive_updates = 0;
    //n->minWeight = 0;

    tree.push_back(n);
    if(label >= 0) tree_labels[n->label] = n;
    if(parent != nullptr) parent->children.push_back(n);
    return n;
}


// public
//----------------------------------------------------------------------------------------------------------------------

real PLT::loss(const std::vector<int32_t>& labels, real lr, Model *model_) {

    std::unordered_set<NodePLT*> n_positive; // positive nodes
    std::unordered_set<NodePLT*> n_negative; // negative nodes

    // PLT's node selection
    if (labels.size() > 0) {
        for (uint32_t i = 0; i < labels.size(); ++i) {
            const auto& ni = tree_labels.find(labels[i]);
            if(ni == tree_labels.end()) throw std::runtime_error("Missing label in the tree structure!");
            NodePLT *n = ni->second;
            n_positive.insert(n);
            while (n->parent) {
                n = n->parent;
                n_positive.insert(n);
            }
        }

        std::queue<NodePLT*> n_queue; // nodes queue
        n_queue.push(tree_root); // push root

        while(!n_queue.empty()) {
            NodePLT* n = n_queue.front(); // current node index
            n_queue.pop();

            if (n->label < 0) {
                for(auto child : n->children) {
                    if (n_positive.count(child)) n_queue.push(child);
                    else n_negative.insert(child);
                }
            }
        }
    }
    else n_negative.insert(tree_root);

    // PLT's negative sampling
    if(args_->neg > 0){
        int n_sampled = 0, n_labels = 0;
        std::priority_queue<NodeProb, std::vector<NodeProb>, std::less<NodeProb>> n_queue;
        n_queue.push({tree_root, predictNode(tree_root, model_->hidden_, model_)});
        while(n_sampled < args_->neg) {
            //while(n_labels < labels.size()) { // alternative negative sampling
            //for(int i = 0; i < args_->neg; ++i) { // alternative negative sampling
            NodePLT *n = getNextBest(n_queue, model_->hidden_, model_).node;
            if(!n_positive.count(n)){
                ++n_sampled;
                while (n->parent) {
                    if (!n_positive.count(n)) n_negative.insert(n);
                    else break;
                    n = n->parent;
                }
            } else ++n_labels;
        }
    }

    real loss = 0.0;
    real l2 = args_->l2;

    real label = 1.0;
    for (auto &n : n_positive){
        loss += learnNode(n, label, lr, l2, model_);
    }

    label = 0.0;
    for (auto &n : n_negative){
        loss += learnNode(n, label, lr, l2, model_);
    }

    y_count += labels.size();
    ++x_count;

    return loss;
}

NodeProb PLT::getNextBest(std::priority_queue<NodeProb, std::vector<NodeProb>, std::less<NodeProb>>& n_queue,
                           Vector& hidden, const Model *model_){

    // while (!n_queue.empty()) {
    while (true) {
        NodeProb np = n_queue.top(); // current node
        n_queue.pop();

        if(!args_->probNorm) {
            if (np.node->children.size()){
                for (auto& child : np.node->children)
                    n_queue.push({child, np.prob * predictNode(child, hidden, model_)});
            }
            if (np.node->label >= 0) return np;
        } else {
            if (np.node->children.size()){
                real sumOfP = 0.0;
                std::vector<NodeProb> normChildren;
                for (auto& child : np.node->children) {
                    real p = predictNode(child, hidden, model_);
                    normChildren.push_back({child, p});
                    sumOfP += p;
                }
                if (sumOfP < 1.0){ //&& (sumOfP > 10e-6)) {
                    for (auto& child : normChildren) {
                        child.prob = child.prob / sumOfP;
                    }
                }
                for (auto& child : normChildren){
                    child.prob *= np.prob;
                    n_queue.push(child);
                }
            }
            if (np.node->label >= 0) return np;
        }

        if(n_queue.empty()) return np;
    }
}

void PLT::findKBest(int32_t top_k, real threshold, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_) {
    std::priority_queue<NodeProb, std::vector<NodeProb>, std::less<NodeProb>> n_queue;
    n_queue.push({tree_root, predictNode(tree_root, hidden, model_)});

    for(int i = 0; i < top_k; ++i) {
        NodeProb np = getNextBest(n_queue, hidden, model_);
        if(np.prob < threshold) break;
        heap.push_back({np.prob, np.node->label});
    }
}

real PLT::getLabelP(int32_t label, Vector &hidden, const Model *model_){

    std::vector<NodePLT*> path;
    NodePLT *n = tree_labels[label];

    if(!args_->probNorm){
        real p = predictNode(n, hidden, model_);
        while(n->parent) {
            n = n->parent;
            p = p * predictNode(n, hidden, model_);
        }
        assert(n == tree_root);
        return p;

    } else {
        path.push_back(n);
        while (n->parent) {
            n = n->parent;
            path.push_back(n);
        }

        assert(tree_root == n);
        assert(tree_root == path.back());

        real p = predictNode(tree_root, hidden, model_);
        for (auto n = path.rbegin(); n != path.rend(); ++n) {
            if ((*n)->label < 0) {
                //TODO: rewrite this part
                /*
                for (auto child : (*n)->children) {
                    normChildren.push_back({child, })
                    child->p = cp * predictNode(child, hidden, model_);
                    sumOfP += child->p;
                }
                if ((sumOfP < cp) //&& (sumOfP > 10e-6)) {
                    for (auto child : (*n)->children) {
                        child->p = (child->p * cp) / sumOfP;
                    }
                }
                float sumOfP = 0.0f;
                 */
            }
        }

        return p;
    }
}

void PLT::setup(std::shared_ptr<Dictionary> dict, uint32_t seed){
  rng.seed(seed);

  if(args_->treeStructure != ""){
    args_->treeType = tree_type_name::custom;
    loadTreeStructure(args_->treeStructure, dict);
    return;
  }

  if (args_->treeType == tree_type_name::complete)
    buildCompletePLTree(dict->nlabels());
  else if (args_->treeType == tree_type_name::huffman)
    buildHuffmanPLTree(dict->getCounts(entry_type::label));
  else if (args_->treeType == tree_type_name::kmeans)
    buildKMeansPLTree(args_, dict);
}

int32_t PLT::getSize(){
  assert(t == tree.size());
  return tree.size();
}

void PLT::printInfo(){
  /*
  std::cerr << "  Avg n vis: " << static_cast<float>(n_vis_count) / x_count << "\n";
  std::cerr << "  Avg n in vis: " << static_cast<float>(n_in_vis_count) / x_count << "\n";
  std::cerr << "  Avg y: " << static_cast<float>(y_count) / x_count << "\n";
  */
}

void PLT::save(std::ostream& out){
    if(args_->verbose > 2)
        std::cerr << "Saving PLT output ...\n";

    out.write((char*) &k, sizeof(int32_t));
    out.write((char*) &shift, sizeof(shift));

    t = tree.size();
    out.write((char*) &t, sizeof(t));
    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = tree[i];
        out.write((char*) &n->index, sizeof(n->index));
        out.write((char*) &n->label, sizeof(n->label));
        out.write((char*) &n->n_updates, sizeof(n->n_updates));
        out.write((char*) &n->n_positive_updates, sizeof(n->n_positive_updates));
    }

    uint32_t root_n = tree_root->index;
    out.write((char*) &root_n, sizeof(root_n));

    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = tree[i];

        int parent_n;
        if(n->parent) parent_n = n->parent->index;
        else parent_n = -1;

        out.write((char*) &parent_n, sizeof(parent_n));
    }
}

void PLT::load(std::istream& in){
    if(args_->verbose > 2)
        std::cerr << "Loading PLT output ...\n";

    in.read((char*) &k, sizeof(int32_t));
    in.read((char*) &shift, sizeof(shift));

    in.read((char*) &t, sizeof(t));
    tree.resize(t);
    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = new NodePLT();
        in.read((char*) &n->index, sizeof(n->index));
        in.read((char*) &n->label, sizeof(n->label));
        in.read((char*) &n->n_updates, sizeof(n->n_updates));
        in.read((char*) &n->n_positive_updates, sizeof(n->n_positive_updates));

        tree[i] = n;
        if (n->label >= 0) tree_labels[n->label] = n;
    }

    uint32_t root_n;
    in.read((char*) &root_n, sizeof(root_n));
    tree_root = tree[root_n];

    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = tree[i];

        int parent_n;
        in.read((char*) &parent_n, sizeof(parent_n));
        if(parent_n >= 0) {
            tree[parent_n]->children.push_back(n);
            n->parent = tree[parent_n];
        }
    }

    if(args_->verbose > 2)
        std::cerr << "  Nodes: " << tree.size() << ", leaves: " << tree_labels.size() << "\n";
}

}
