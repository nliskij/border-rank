#include "flipwalk.hpp"
#include <tuple>
#include <utility>
#include <iostream>
#include <cassert>

constexpr Decomp elem_decomp() {
    Decomp d(R);
    int l = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                Mat1 m1;
                Mat2 m2;
                Mat3 m3;
                m1.set(N * i + j);
                m2.set(P * j + k);
                m3.set(P * i + k);
                d[l] = std::make_tuple(m1, m2, m3);
                l++;
            }
        }
    }
    return d;
}


void FlipWalk::set_decomp(Decomp d) {
    this->decomp = d;
    // this change actually gives a noticeable speed up
    val_to_idx1.max_load_factor(0.5f);
    val_to_idx2.max_load_factor(0.5f);
    val_to_idx3.max_load_factor(0.5f);

    for (int i = 0; i < this->decomp.size(); i++) {
        if (not(this->val_to_idx1.contains(std::get<U>(d[i]))))
            this->val_to_idx1[std::get<U>(d[i])] = SamplingSet<std::size_t>();
        this->val_to_idx1[std::get<U>(d[i])].insert(i);
    }
    for (int i = 0; i < this->decomp.size(); i++) {
       if (not(this->val_to_idx2.contains(std::get<V>(d[i]))))
            this->val_to_idx2[std::get<V>(d[i])] = SamplingSet<std::size_t>();
        this->val_to_idx2[std::get<V>(d[i])].insert(i);
    }
    for (int i = 0; i < this->decomp.size(); i++) {
        if (not(this->val_to_idx3.contains(std::get<W>(d[i]))))
            this->val_to_idx3[std::get<W>(d[i])] = SamplingSet<std::size_t>();
        this->val_to_idx3[std::get<W>(d[i])].insert(i);
    }
    for (const auto& pair : this->val_to_idx1) {
        if (std::size(pair.second) > 1)
            this->repeats1.insert(pair.first);
    }
    for (const auto& pair : this->val_to_idx2) {
        if (std::size(pair.second) > 1)
            this->repeats2.insert(pair.first);
    }
    for (const auto& pair : this->val_to_idx3) {
        if (std::size(pair.second) > 1)
            this->repeats3.insert(pair.first);
    }
    this->rank = d.size();
}

FlipWalk::FlipWalk() {
    this->set_decomp(elem_decomp());
}

std::tuple<bool, Col, int>  FlipWalk::apply_flip(Flip f) {
    Mat1& u1 = std::get<U>(decomp[f.fst]);
    Mat1& u2 = std::get<U>(decomp[f.snd]);
    Mat2& v1 = std::get<V>(decomp[f.fst]);
    Mat2& v2 = std::get<V>(decomp[f.snd]);
    Mat3& w1 = std::get<W>(decomp[f.fst]);
    Mat3& w2 = std::get<W>(decomp[f.snd]);

    // this is repetetive, presumably it could be cleaned up with templates
    if (f.col == U) {
        //assert(u1 == u2);
        Mat2 new_fst = v1 ^ v2;
        Mat3 new_snd = w1 ^ w2;

        this->val_to_idx2[v1].erase(f.fst);
        this->val_to_idx3[w2].erase(f.snd);

        if (std::size(val_to_idx2[v1]) < 2)
            this->repeats2.erase(v1);
        if (std::size(val_to_idx3[w2]) < 2)
            this->repeats3.erase(w2);
        // for small sizes, there is a big boost from not erasing entries with empty sets
        // I guess this is because we don't malloc/free excessively.
        // but for large sizes, this slows down! Maybe because of hash collisions?
        // best fix would be to use an object pool since at most 3 * rank SamplingSets are needed at a time, no need to malloc/free
        if (std::size(val_to_idx2[v1]) == 0)
            this->val_to_idx2.erase(v1);
        if (std::size(val_to_idx3[w2]) == 0)
            this->val_to_idx3.erase(w2);
        this->val_to_idx2.try_emplace(new_fst);
        this->val_to_idx3.try_emplace(new_snd);
        this->val_to_idx2[new_fst].insert(f.fst);
        this->val_to_idx3[new_snd].insert(f.snd);

        if (std::size(this->val_to_idx2[new_fst]) > 1)
            this->repeats2.insert(new_fst);
        if (std::size(this->val_to_idx3[new_snd]) > 1)
            this->repeats3.insert(new_snd);

        v1 = new_fst;
        w2 = new_snd;

        // not sure if the unlikely is doing anything
        if (new_fst == 0) [[unlikely]]
            return {true, V, f.fst};
        if (new_snd == 0) [[unlikely]]
            return {true, W, f.snd};
    }
    else if (f.col == V) {
        //assert(v1 == v2);
        Mat1 new_fst = u1 ^ u2;
        Mat3 new_snd = w1 ^ w2;

        this->val_to_idx1[u1].erase(f.fst);
        this->val_to_idx3[w2].erase(f.snd);

        if (std::size(val_to_idx1[u1]) < 2)
            this->repeats1.erase(u1);
        if (std::size(val_to_idx3[w2]) < 2)
            this->repeats3.erase(w2);
        if (std::size(val_to_idx1[u1]) == 0)
            this->val_to_idx1.erase(u1);
        if (std::size(val_to_idx3[w2]) == 0)
            this->val_to_idx3.erase(w2);
        this->val_to_idx1.try_emplace(new_fst);
        this->val_to_idx3.try_emplace(new_snd);
        this->val_to_idx1[new_fst].insert(f.fst);
        this->val_to_idx3[new_snd].insert(f.snd);

        if (std::size(this->val_to_idx1[new_fst]) > 1)
            this->repeats1.insert(new_fst);
        if (std::size(this->val_to_idx3[new_snd]) > 1)
            this->repeats3.insert(new_snd);

        u1 = new_fst;
        w2 = new_snd;

        if (new_fst == 0) [[unlikely]]
            return {true, U, f.fst};
        if (new_snd == 0) [[unlikely]]
            return {true, W, f.snd};
    }
    else if (f.col == W) {
        //assert(w1 == w2);
        Mat1 new_fst = u1 ^ u2;
        Mat2 new_snd = v1 ^ v2;

        this->val_to_idx1[u1].erase(f.fst);
        this->val_to_idx2[v2].erase(f.snd);

        if (std::size(val_to_idx1[u1]) < 2)
            this->repeats1.erase(u1);
        if (std::size(val_to_idx2[v2]) < 2)
            this->repeats2.erase(v2);
        if (std::size(val_to_idx1[u1]) == 0)
            this->val_to_idx1.erase(u1);
        if (std::size(val_to_idx2[v2]) == 0)
            this->val_to_idx2.erase(v2);
        this->val_to_idx1.try_emplace(new_fst);
        this->val_to_idx2.try_emplace(new_snd);
        this->val_to_idx1[new_fst].insert(f.fst);
        this->val_to_idx2[new_snd].insert(f.snd);

        if (std::size(this->val_to_idx1[new_fst]) > 1)
            this->repeats1.insert(new_fst);
        if (std::size(this->val_to_idx2[new_snd]) > 1)
            this->repeats2.insert(new_snd);

        u1 = new_fst;
        v2 = new_snd;

        if (new_fst == 0) [[unlikely]]
            return {true, U, f.fst};
        if (new_snd == 0) [[unlikely]]
            return {true, V, f.snd};
    }

    return {false, NoFlip, -1};
}

Flip FlipWalk::sample_flip() {
    const std::size_t s1 = repeats1.size();
    const std::size_t s2 = repeats2.size();
    const std::size_t s3 = repeats3.size();
    const std::size_t total = s1 + s2 + s3;

    if (total == 0)
        return {NoFlip, 0, 0};

    // note that my sampling is not uniform over flips.
    // this code samples a column, weighted by the number of repeated values
    // in that column. within a column, a repeated value is sampled uniformly,
    // but this may not be uniform over all possible flips.
    // e.g., if column1 is val1, val1, val1, val2, val2, then the possible flips are:
    // (0, 1), (0, 2), (1, 2), (3, 4)
    // but I chooose between val1 and val2 with prob 0.5 each, then sample a flip from that value.
    // this is only for performance reasons, and experimentation shows no harm
    // compared to truly uniform sampling.
    std::uniform_int_distribution<std::size_t> pick(0, total - 1);
    std::size_t r = pick(rng);

    if (r < s1) {
        auto key = repeats1.sample1();
        auto p = val_to_idx1[key].sample2_distinct();
        return {U, p.first, p.second};
    } 
    else if (r < s1 + s2) {
        auto key = repeats2.sample1();
        auto p = val_to_idx2[key].sample2_distinct();
        return {V, p.first, p.second};
    } 
    else {
        auto key = repeats3.sample1();
        auto p = val_to_idx3[key].sample2_distinct();
        return {W, p.first, p.second};
    }
}

int FlipWalk::run(int steps) {
    for (int step = 0; step < steps; step++) {
        Flip f = this->sample_flip();
        if (f.col == NoFlip) {
            std::cout << "no flips" << std::endl;
            return step;
        }
        auto [reduced, col, idx] = this->apply_flip(f);
        if (reduced) {
            this->decomp.erase(decomp.begin() + idx);
            this->val_to_idx1.clear();
            this->val_to_idx2.clear();
            this->val_to_idx3.clear();
            this->repeats1.clear();
            this->repeats2.clear();
            this->repeats3.clear();
            this->set_decomp(this->decomp);
            std::cout << "reduced to rank: " << this->rank << " on step " << step << std::endl;
        }
    }
    return steps;
}

void FlipWalk::dump() {
    for (auto [u, v, w] : this->decomp) {
        std::cout << u.to_ulong() << "|" << v.to_ulong() << "|" << w.to_ulong() << std::endl;
    }
    std::cout << std::endl;
}

Decomp str_to_decomp(int rank, std::vector<std::string>& str) {
    constexpr std::size_t S1 = M * N;
    constexpr std::size_t S2 = N * P;
    constexpr std::size_t S3 = M * P;

    if (str.size() != static_cast<std::size_t>(3 * rank)) {
        throw std::invalid_argument("str_to_decomp: input vector size must be exactly 3 * rank");
    }

    auto check = [](const std::string& s, std::size_t expected, const char* name) {
        if (s.size() != expected) {
            throw std::invalid_argument(
                std::string("str_to_decomp: wrong length for ") + name +
                " (expected " + std::to_string(expected) +
                ", got " + std::to_string(s.size()) + ")"
            );
        }
        if (s.find_first_not_of("01") != std::string::npos) {
            throw std::invalid_argument(
                std::string("str_to_decomp: non-binary characters found in ") + name
            );
        }
    };

    Decomp out;
    out.reserve(static_cast<std::size_t>(rank));

    for (int i = 0; i < rank; ++i) {
        const std::string& s1 = str[static_cast<std::size_t>(i)];
        const std::string& s2 = str[static_cast<std::size_t>(rank + i)];
        const std::string& s3 = str[static_cast<std::size_t>(2 * rank + i)];

        check(s1, S1, "Mat1");
        check(s2, S2, "Mat2");
        check(s3, S3, "Mat3");

        out.emplace_back(Mat1{s1}, Mat2{s2}, Mat3{s3});
    }

    return out;
}

int main(int argc, char* argv[]) {
    int steps = std::stoi(argv[1]);
    FlipWalk walk = FlipWalk();
    if (argc > 2) {
        int rank = std::stoi(argv[2]);
        std::vector<std::string> strs(argv + 2, argv + 2 + 3 * rank);
        walk.set_decomp(str_to_decomp(rank, strs));
    }
    int ran = walk.run(steps);
    std::cout << "steps completed: " << ran << std::endl;
    return 0;
}
