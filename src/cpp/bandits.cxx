//
// Created by Cyrille Kone on 07/01/2023.
//
#include <vector>
#include <random>
#include "utils.hpp"
#include "bandits.hpp"
#include <algorithm>
#include <cassert>
void bandit::reset_env(size_t seed) {
    gen.seed(seed);
    this->seed = seed;
}
bandit::bandit(const std::vector<std::vector<double>>& arms_means){
    this->arms_means = arms_means;
    K = arms_means.size();
    D = arms_means[0].size();
    action_space.resize(K);
    std::iota(action_space.begin(), action_space.end(), size_t{0});
    optimal_arms_mask = std::move(pareto_optimal_arms_mask(arms_means));
    std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(optimal_arms), [&](size_t i){
        return optimal_arms_mask[i];
    });
    // sort the list of arms in ascending order
    std::sort(optimal_arms.begin(), optimal_arms.end());
    // initialize the generator with default seed
    gen = std::mt19937(seed);
    // compute H
    suboptimal_gaps = compute_gap(optimal_arms_mask, arms_means);
    H = std::accumulate(suboptimal_gaps.begin(), suboptimal_gaps.end(),0., [](double sum, double x){
        return sum + 1./std::pow(x, 2);
    });
}
bernoulli::bernoulli(const std::vector<std::vector<double>> & arms_means): bandit(arms_means){
    sigma = 0.5;
}
std::vector<std::vector<double>> bernoulli::sample(const std::vector<size_t> &arms)  {
    std::vector<std::vector<double>> _val(arms.size());
    std::vector<double> _v(D);
    std::generate(_val.begin(), _val.end(), [&, k=0]() mutable {
        std::generate(_v.begin(), _v.end(), [&, d=0] () mutable {
            return std::bernoulli_distribution(arms_means[arms[k]][d++])(gen);
        });
        ++k;
        return _v;
    });
        return _val;
}
std::vector<std::vector<double>> bernoulli::sample(const std::vector<size_t> &&arms)  {
    std::vector<std::vector<double>> _val(arms.size());
    std::vector<double> _v(D);
    std::generate(_val.begin(), _val.end(), [&, k=0]() mutable {
        std::generate(_v.begin(), _v.end(), [&, d=0] () mutable {
            return std::bernoulli_distribution(arms_means[arms[k]][d++])(gen);
        });
        ++k;
        return _v;
    });
    return _val;
}

gaussian::gaussian(const std::vector<std::vector<double>> &arms_means, const std::vector<double>& stddev): bandit(arms_means) {
    assert(stddev.size() == arms_means[0].size());
    this->stddev = stddev;
    sigma = 1.;
}
std::vector<std::vector<double>> gaussian::sample(const std::vector<size_t>& arms) {
    std::vector<std::vector<double>> _val(arms.size());
    std::vector<double> _v(D);
    std::generate(_val.begin(), _val.end(), [&, k=0]() mutable {
        std::generate(_v.begin(), _v.end(), [&] ()  {
            return std::normal_distribution(0.)(gen);
        });
        return sum(prod(stddev, _v), arms_means[arms[k++]]) ;
    });
    return _val;
}

std::vector<std::vector<double>> gaussian::sample(const std::vector<size_t>&& arms) {
    std::vector<std::vector<double>> _val(arms.size());
    std::vector<double> _v(D);
    std::generate(_val.begin(), _val.end(), [&, k=0]() mutable {
        std::generate(_v.begin(), _v.end(), [&] ()  {
            return std::normal_distribution(0.)(gen);
        });
        return sum(prod(stddev, _v), arms_means[arms[k++]]) ;
    });
    return _val;
}