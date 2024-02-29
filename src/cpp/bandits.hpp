//
// Created by Cyrille Kone on 07/01/2023.
//

#pragma once
#include <random>
#include <vector>
#include <cstddef>
thread_local inline std::mt19937 gen;
struct bandit{
    size_t K;
    size_t D;
    double H;
    size_t seed{42};
    double sigma;
    //std::mt19937 gen;
    size_t optimal_arm; //  only defined for 1d bandit
    std::vector<double> suboptimal_gaps;
    std::vector<size_t> action_space;
    std::vector<size_t> optimal_arms;
    std::vector<bool> optimal_arms_mask;
    std::vector<std::vector<double>> arms_means;
    std::vector<std::vector<double>> cov;
    bandit() = default;
    explicit bandit(const std::vector<std::vector<double>>&);
    virtual std::vector<std::vector<double>> sample(const std::vector<size_t>&)=0;
    virtual std::vector<std::vector<double>> sample(const std::vector<size_t> &&)=0;
    void reset_env(size_t);
};
struct bernoulli: bandit{
    bernoulli() = default;
    explicit bernoulli(const std::vector<std::vector<double>>&);
    std::vector<std::vector<double>> sample(const std::vector<size_t> &) override;
    std::vector<std::vector<double>> sample(const std::vector<size_t> &&) override;
};

struct gaussian: bandit {
    gaussian() = default;
    gaussian(const std::vector<std::vector<double>>&, const std::vector<double>&);
    std::vector<std::vector<double>> sample(const std::vector<size_t> &) override;
    std::vector<std::vector<double>> sample(const std::vector<size_t> &&) override;
    std::vector<double> stddev;
};