#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <chrono>
#include <thread>
#include <functional>

// trying a 'fast' rng is probably pointless, but I tried it out
static inline uint64_t splitmix64(uint64_t &x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct xoroshiro128plus {
    uint64_t s0, s1;
    static inline uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
    explicit xoroshiro128plus(uint64_t seed = 0x853c49e6748fea9bULL) noexcept {
        uint64_t x = seed;
        s0 = splitmix64(x);
        s1 = splitmix64(x);
    }
    inline uint64_t next_u64() noexcept {
        uint64_t r = s0 + s1;
        uint64_t t = s1 ^ s0;
        s0 = rotl(s0, 55) ^ t ^ (t << 14);
        s1 = rotl(t, 36);
        return r;
    }
    inline uint32_t next_bounded(uint32_t bound) noexcept {
        uint64_t x = next_u64();
    #if defined(__SIZEOF_INT128__)
        __uint128_t m = (__uint128_t)x * bound;
        uint32_t l = (uint32_t)m;
        if (l < bound) {
            uint32_t t = (uint32_t(-int32_t(bound)) % bound);
            while (l < t) {
                x = next_u64();
                m = (__uint128_t)x * bound;
                l = (uint32_t)m;
            }
        }
        return (uint32_t)(m >> 64);
    #else
        uint64_t m = x * uint64_t(bound);
        uint32_t l = (uint32_t)m;
        if (l < bound) {
            uint32_t t = (uint32_t(-int32_t(bound)) % bound);
            while (l < t) {
                x = next_u64();
                m = x * uint64_t(bound);
                l = (uint32_t)m;
            }
        }
        return (uint32_t)(m >> 32);
    #endif
    }
};

inline xoroshiro128plus& tls_rng() noexcept {
    static thread_local xoroshiro128plus rng([]() noexcept {
        uint64_t base = 0x12345678abcdef00ULL ^
                        (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
        uint64_t x = base ^ (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return splitmix64(x);
    }());
    return rng;
}

// Transparent hash/equal so we can look up by uint64_t etc. when T is integral-like.
template <class T>
struct transparent_hash {
    using is_transparent = void; // enables heterogeneous lookup
    size_t operator()(const T& x) const noexcept { return std::hash<T>{}(x); }
    template <class U, std::enable_if_t<!std::is_same_v<std::decay_t<U>, T>, int> = 0>
    size_t operator()(const U& x) const noexcept { return std::hash<std::decay_t<U>>{}(x); }
};

template <class T>
struct transparent_equal {
    using is_transparent = void;
    template <class A, class B>
    bool operator()(const A& a, const B& b) const noexcept { return a == b; }
};

template <class T>
class SamplingSet {
public:
    using size_type = std::size_t;

    SamplingSet() noexcept { index_.max_load_factor(0.5f); }
    explicit SamplingSet(size_type reserve_hint) {
        reserve(reserve_hint);
        index_.max_load_factor(0.5f);
    }

    SamplingSet(std::initializer_list<T> init) {
        reserve(init.size());
        index_.max_load_factor(0.5f);
        for (const auto &v : init) insert(v);
    }

    template <class U>
    bool insert(U&& value) {
        // single lookup + strong exception safety
        const size_type pos = data_.size();
        auto [it, inserted] = index_.try_emplace(value, pos);
        if (!inserted) return false;
        try {
            data_.push_back(std::forward<U>(value));
        } catch (...) {
            index_.erase(it); // rollback
            throw;
        }
        return true;
    }

    bool erase(const T& value) noexcept {
        auto it = index_.find(value);
        if (it == index_.end()) return false;
        size_type i = it->second;
        size_type last = data_.size() - 1;
        if (i != last) {
            std::swap(data_[i], data_[last]);
            index_[data_[i]] = i; // use operator[] with existing key (no extra lookup in practice)
        }
        data_.pop_back();
        index_.erase(it);
        return true;
    }

    template <class K>
    bool erase_key(const K& k) noexcept { // heterogeneous erase (optional)
        auto it = index_.find(k);
        if (it == index_.end()) return false;
        return erase(data_[it->second]);
    }

    [[nodiscard]] bool contains(const T& value) const noexcept {
        return index_.find(value) != index_.end();
    }

    [[nodiscard]] size_type size()  const noexcept { return data_.size(); }
    [[nodiscard]] bool      empty() const noexcept { return data_.empty(); }

    void reserve(size_type n) {
        data_.reserve(n);
        index_.reserve(n);
    }

    void clear() noexcept {
        data_.clear();
        index_.clear();
    }

    [[nodiscard]] const T& sample1() const {
        if (data_.empty()) throw std::runtime_error("sample1 on empty set");
        uint32_t i = tls_rng().next_bounded((uint32_t)data_.size());
        return data_[i];
    }

    [[nodiscard]] std::pair<const T&, const T&> sample2_distinct() const {
        const size_type n = data_.size();
        if (n < 2) throw std::runtime_error("sample2_distinct needs size >= 2");
        auto& rng = tls_rng();
        uint32_t i  = rng.next_bounded((uint32_t)n);
        uint32_t j0 = rng.next_bounded((uint32_t)(n - 1));
        uint32_t j  = j0 + (j0 >= i);
        return { data_[i], data_[j] };
    }

    [[nodiscard]] const std::vector<T>& data() const noexcept { return data_; }

private:
    std::vector<T> data_;
    std::unordered_map<T, size_type, transparent_hash<T>, transparent_equal<T>> index_;
};

