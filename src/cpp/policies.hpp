#pragma once
#include<cstddef>
#include<iostream>
#include "utils.hpp"
#include "bandits.hpp"
struct policy_fb{
    size_t K;
    size_t dim;
    size_t D;
    double sigma;
    std::vector<size_t> action_space;
    bandit* bandit_ref;
    policy_fb() = default;
    explicit policy_fb( bandit&);
    std::pair<std::pair<size_t, bool>, std::vector<size_t>> loop(const size_t seed, const size_t T);
};


struct ege_sr: policy_fb {
    ege_sr() = default;
    explicit ege_sr(bandit&);
    [[nodiscard]] bool loop(const size_t seed, const size_t T, const size_t k);
};

struct ege_srk: policy_fb {
    ege_srk() = default;
    explicit ege_srk(bandit&);
    [[nodiscard]] std::pair<bool, std::pair<std::vector<size_t>, std::pair<size_t, size_t>>> loop(const size_t seed, const size_t T, const size_t k);
};

struct ege_sh: policy_fb {
    ege_sh() = default;
    explicit ege_sh(bandit&);
    [[nodiscard]] bool loop(const size_t seed, const size_t T);
};

struct ua: policy_fb {
    ua() = default;
    explicit ua(bandit&);
    [[nodiscard]] bool loop(const size_t seed, const size_t T);
};
std::vector<double> batch_sr(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds,size_t k );
std::pair<std::vector<double>, std::pair<std::vector<std::vector<std::vector<size_t>>>, std::pair<std::vector<double>, std::vector<double>>>> batch_srk(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds,size_t k );
std::vector<double> batch_ua(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds);
std::vector<double> batch_sh(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds);