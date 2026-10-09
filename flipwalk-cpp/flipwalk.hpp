#include <vector>
#include <tuple>
#include <bitset>
#include <cstddef>
#include <random>
#include <string>

#include "sampling.hpp"

#ifndef M
#define M 4
#endif
#ifndef N
#define N 3
#endif
#ifndef P
#define P 4
#endif
#define R M * N * P

typedef std::bitset<M * N> Mat1;
typedef std::bitset<N * P> Mat2;
typedef std::bitset<M * P> Mat3;

typedef std::vector<std::tuple<Mat1, Mat2, Mat3>> Decomp;
constexpr Decomp elem_decomp();
Decomp str_to_decomp(int rank, std::vector<std::string>& str);

// maybe identity is a silly hash, but it outperforms whatever the default is
// I think hash collisions are a non-negligible cost, so worth thinking about
struct Mat1Hash {
    std::size_t operator()(const Mat1& b) const noexcept {
        if constexpr (M * N <= 64)
            return b.to_ullong();
        else
            return std::hash<Mat1>()(b);
    }
};
struct Mat2Hash {
    std::size_t operator()(const Mat2& b) const noexcept {
        if constexpr (N * P <= 64)
            return b.to_ullong();
        else
            return std::hash<Mat2>()(b);
    }
};
struct Mat3Hash {
    std::size_t operator()(const Mat3& b) const noexcept {
        if constexpr (M * P <= 64)
            return b.to_ullong();
        else
            return std::hash<Mat3>()(b);
    }
};

enum Col {U, V, W, NoFlip};
struct Flip {
    Col col;
    std::size_t fst;
    std::size_t snd;
};

class FlipWalk {
public:
    Decomp decomp;
    FlipWalk();
    FlipWalk(Decomp d);
    void set_decomp(Decomp d);
    std::tuple<bool, Col, int> apply_flip(Flip f);
    Flip sample_flip();
    int run(int steps);
    void dump();
    std::size_t rank;
private:
    // SamplingSet is a set with O(1) random sampling
    std::unordered_map<Mat1, SamplingSet<size_t>, Mat1Hash> val_to_idx1;
    std::unordered_map<Mat2, SamplingSet<size_t>, Mat2Hash> val_to_idx2;
    std::unordered_map<Mat3, SamplingSet<size_t>, Mat3Hash> val_to_idx3;
    SamplingSet<Mat1> repeats1;
    SamplingSet<Mat2> repeats2;
    SamplingSet<Mat3> repeats3;
    std::mt19937_64 rng{std::random_device{}()};
};
