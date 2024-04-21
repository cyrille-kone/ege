#pragma once
#include<vector>
#include<cstddef>
#include <numeric>
#include<iostream>
#include<cmath>
#include <algorithm>

#define INF 1e12
// maximum number of arms that can be handled
#define MAX_K 100000
#define EE 2.71828182845904523536
#define get_argmin(v, idx) (idx)[std::distance((v).begin(), std::min_element((v).begin(), (v).end()))]
#define get_argmax(v, idx) (idx)[std::distance((v).begin(), std::max_element((v).begin(), (v).end()))]
#define in_set(id, v) (std::find((v).begin(), (v).end(), (id)) != (v).end())
inline double Cg(const double& delta){
    return log(1/delta)/2 + std::max(log(log(1/delta)/2), 0.);
}
std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>>&means);
std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>> &means, const std::vector<bool>& active_mask);
// using namespace std;
// utility functions for arrays
std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>> &means_t, const std::vector<double>& betas, double eps);
inline double minimum_quantity_dom(const std::vector<double>& xi, const std::vector<double>& xj, const double eps){
    double res{-INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::max(res, xi[k] + eps - xj[k]);
    return res;
}
// @utility function
inline double M(const std::vector<double>& xi, const std::vector<double>& xj) {
    double res{-INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::max(res, xi[k]  - xj[k]);
    return res;
}
// @utility function
inline double m(const std::vector<double>& xi, const std::vector<double>& xj){
    double res{INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::min(res, xj[k] - xi[k]);
    return res;
}
//@utility function
double sub_opt_gap(size_t i, const std::vector<std::vector<double>>& means, const std::vector<double>& vec_delta_star, const std::vector<bool>& active_mask, bool par);
// @utility function
std::vector<double> delta_star(const std::vector<std::vector<double>>& means, const std::vector<bool>& active_mask);
//@utility function
template <typename T>
inline std::vector<T> sum(const std::vector<T>& xi, const std::vector<T>& xj){
    // assert same size?
    size_t n{xi.size()};
    std::vector<T> res;
    res.reserve(n);
    std::transform(xi.begin(), xi.end(), xj.begin(), std::back_inserter(res), [](T x, T y){return x+y;});
    return res;
}


inline double minimum_quantity_non_dom(const std::vector<double> & xi, const std::vector<double>& xj, const double eps){
    double res{INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::min(res, xj[k] - xi[k] + eps );
    return res;
}
/*
 * Return true if xi is dominated by xj
 */
inline bool is_pareto_dominated(const std::vector<double>& xi, const std::vector<double>& xj, const double& eps){
    bool is_strict{false};
    for (size_t k{0}; k<xi.size(); ++k){
        if (xi[k] + eps > xj[k]) return false;
        is_strict |= (xi[k] + eps < xj[k]);
    }
    return is_strict;
};
std::vector<double> compute_gap(const std::vector<bool>& pareto_set_mask, const std::vector<std::vector<double>>&arms_means);




// debugging
template <typename T>
void print_array1d(std::vector<T>& vec){
    std::cout<<"[ ";
    for (auto t: vec)  std::cout<<t<<", ";
    std::cout<<"]"<<std::endl;
}
template<typename  T>
void print_array2d(std::vector<std::vector<T>>& vec){
    std::cout<<"[ ";
    for (auto t: vec) print_array1d(t);
    std::cout<<"]"<<std::endl;
}
template <typename T>
void print_array1d(const std::vector<T>& vec){
    std::cout<<"[ ";
    for (auto t: vec)  std::cout<<t<<", ";
    std::cout<<"]"<<std::endl;
}
template<typename  T>
void print_array2d(const std::vector<std::vector<T>>& vec){
    std::cout<<"[ ";
    for (auto t: vec) print_array1d(t);
    std::cout<<"]"<<std::endl;
}

template <typename T>
inline std::vector<T> prod(const std::vector<T> xi, const std::vector<T> xj){
    //assert(xi.size()== xj.size());
    std::vector<T> res(xi.size());
    for (size_t i{0}; i<xi.size(); ++i){
        res[i] = xi[i]*  xj[i];
    }
    return res;
}