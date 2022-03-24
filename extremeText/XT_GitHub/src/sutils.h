/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 *
 * Code from napkinXC
 * https://github.com/mwydmuch/napkinXC
 */

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <algorithm>

#include "real.h"
#include "smatrix.h"

namespace fasttext {

struct Feature {
    int index;
    real value;

    bool operator<(const Feature& r) const { return index < r.index; }
    bool operator>(const Feature& r) const { return index > r.index; }

    friend std::ostream& operator<<(std::ostream& os, const Feature& fn) {
        os << fn.index << ":" << fn.value;
        return os;
    }
};

// Sparse utils

// Sparse vector dot dense vector
template <typename T>
inline T dotVectors(Feature* vector1, const T* vector2, const int& size){
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1 && f->index < size) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template <typename T>
inline T dotVectors(Feature* vector1, const T* vector2){ // Version without size checks
    T val = 0;
    Feature* f = vector1;
    while(f->index != -1) {
        val += f->value * vector2[f->index];
        ++f;
    }
    return val;
}

template <typename T>
inline T dotVectors(Feature* vector1, const std::vector<T>& vector2){
    //dotVectors(vector1, vector2.data(), vector2.size());
    dotVectors(vector1, vector2.data());
}

// Sets values of a dense vector to values of a sparse vector
template <typename T>
inline void setVector(Feature* vector1, T* vector2, size_t size, int shift = 0){
    Feature* f = vector1;
    while(f->index != -1 && f->index + shift < size){
        vector2[f->index + shift] = f->value;
        ++f;
    }
}

template <typename T>
inline void setVector(Feature* vector1, T* vector2, int shift = 0){ // Version without size checks
    Feature* f = vector1;
    while(f->index != -1){
        vector2[f->index + shift] = f->value;
        ++f;
    }
}

template <typename T>
inline void setVector(Feature* vector1, std::vector<T>& vector2, int shift = 0) {
    //setVector(vector1, vector2.data(), vector2.size(), shift);
    setVector(vector1, vector2.data(), shift);
}

// Zeros selected values of a dense vactor
template <typename T>
inline void setVectorToZeros(Feature* vector1, T* vector2, size_t size, int shift = 0){
    Feature* f = vector1;
    while(f->index != -1 && f->index + shift < size){
        vector2[f->index + shift] = 0;
        ++f;
    }
}

template <typename T>
inline void setVectorToZeros(Feature* vector1, T* vector2, int shift = 0){ // Version without size checks
    Feature* f = vector1;
    while(f->index != -1){
        vector2[f->index + shift] = 0;
        ++f;
    }
}

template <typename T>
inline void setVectorToZeros(Feature* vector1, std::vector<T>& vector2, int shift = 0) {
    //setVectorToZeros(vector1, vector2.data(), vector2.size());
    setVectorToZeros(vector1, vector2.data(), shift);
}

// Adds values of sparse vector to dense vector
template <typename T>
inline void addVector(Feature* vector1, T* vector2, size_t size){
    Feature* f = vector1;
    while(f->index != -1 && f->index < size){
        vector2[f->index] += f->value;
        ++f;
    }
}

template <typename T>
inline void addVector(Feature* vector1, std::vector<T>& vector2) {
    addVector(vector1, vector2.data(), vector2.size());
}

// Unit norm
template <typename T>
inline void unitNorm(T* data, size_t size){
    T norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f] * data[f];
    norm = std::sqrt(norm);
    if(norm == 0) return;
    else for(int f = 0; f < size; ++f) data[f] /= norm;
}

inline void unitNorm(Feature* data, size_t size){
    real norm = 0;
    for(int f = 0; f < size; ++f) norm += data[f].value * data[f].value;
    norm = std::sqrt(norm);
    if(norm == 0) return;
    else for(int f = 0; f < size; ++f) data[f].value /= norm;
}

template <typename T>
inline void unitNorm(std::vector<T>& vector){
    unitNorm(vector.data(), vector.size());
}

}
