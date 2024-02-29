#include<vector>
#include <algorithm>
#include "utils.hpp"
#include <execution>


std::vector<double> delta_star(const std::vector<std::vector<double>>& means, const std::vector<bool>& active_mask){
    size_t K {means.size()};
    std::vector<double> res(K);
    for (size_t i {0}; i < K; ++i) {
        if (!active_mask[i]){ res[i] = -INF; continue;}
        res[i] = -INF;
        for (size_t j{0}; j <K; ++j) {
            if (!active_mask[j]) continue;
            res[i] = std::max(res[i], m(means[i], means[j]));
        }
    }
    return res;
}

std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>>&means){
    bool is_dom;
    size_t K{means.size()};
    std::vector<bool> res(K);
    for(size_t i{0}; i<K; ++i){
        is_dom = false;
        for(size_t j{0}; j<K; ++j){
            if (i == j) continue;
            is_dom = is_pareto_dominated(means[i], means[j], 0.); // true if mu_i is dominated by mu_j
            if (is_dom) {
                break;}
        }
        res[i] = (!is_dom);
    }
    return res;
}

double get_z1t(const std::vector<std::vector<double>> &means, const std::vector<size_t>& St, const std::vector<std::vector<double>>& beta) {
    if (St.empty()) return INF; // should raise exception
    return std::transform_reduce(St.begin(), St.end(), INF,
                                 [](double xmin, double x){return std::min(xmin,x);},
                                 [&](size_t i){
                                     return get_h(i, means, beta);
                                 });
}
double get_z2t(const std::vector<std::vector<double>> & means, const std::vector<size_t> & St_comp,
               const std::vector<std::vector<double>>& beta) {
    if (St_comp.empty()) return INF;
    return std::transform_reduce(St_comp.begin(), St_comp.end(), INF,
                                 [](double xmin, double x) { return std::min(xmin, x); },
                                 [&](size_t i) {
                                     return std::max(get_h(i, means, beta), get_g(i, means, beta));
                                 });
}


std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>>&mus, const std::vector<double>& betas, double eps=0.){
    bool is_dom;
    size_t K{mus.size()};
    std::vector<bool> res(K);
    for(size_t i{0}; i<K; ++i){
        is_dom = false;
        for(size_t j{0}; j<K; ++j){
            if (i == j) continue;
            is_dom = is_pareto_dominated(mus[i], mus[j], betas[i] - betas[j] + eps); // true if mu_i is dominated by mu_j
            if (is_dom) {
                break;}
        }
        res[i] = (!is_dom);
    }
    return res;
}

std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>>&means, const std::vector<bool>& active_mask){
    bool is_dom;
    size_t K{means.size()};
    std::vector<bool> res(K);
    for(size_t i{0}; i<K; ++i){
        is_dom = false;
        if (!active_mask[i]) continue;
        for(size_t j{0}; j<K; ++j){
            if (i == j || !active_mask[j]) continue;
            is_dom = is_pareto_dominated(means[i], means[j], 0.); // true if mu_i is dominated by mu_j
            if (is_dom) {
                break;}
        }
        res[i] = (!is_dom);
    }
    return res;
}


double sub_opt_gap(size_t i, const std::vector<std::vector<double>>& means, const std::vector<double>& vec_delta_star, const std::vector<bool>& active_mask, bool par) {
    if (!par) return vec_delta_star[i];
    double res{INF};
    size_t K{means.size()};
    for (size_t j{0}; j<K; ++j){
        if (! active_mask[j]) continue;
        res = std::min(res, INF*(i==j) + std::min(M(means[i], means[j]), std::max(M(means[j], means[i]), 0.) + std::max(vec_delta_star[j], 0.)+INF*(i==j)));
    }
    return res;
}


std::vector<double> compute_gap(const std::vector<bool>& pareto_set_mask, const std::vector<std::vector<double>>&arms_means){
    size_t K{pareto_set_mask.size()};
    std::vector<double> deltas(K);
    std::vector<double> tmp(K);
    std::vector<::size_t> arms(K);
    std::vector<size_t> opt_arms;
    std::vector<size_t> subopt_arms;
    double d1, d2, d3; // different of the gap of an optimal arm
    for(size_t k{0}; k<K;++k) arms[k]=k;
    std::copy_if(arms.begin(), arms.end(), std::back_inserter(opt_arms),[&pareto_set_mask](size_t k){return pareto_set_mask[k];} );
    std::copy_if(arms.begin(), arms.end(), std::back_inserter(subopt_arms),[&pareto_set_mask](size_t k){return !pareto_set_mask[k];} );
    for(auto k:subopt_arms){
        std::transform(arms.begin(), arms.end(), tmp.begin(), [&](size_t j){
            return minimum_quantity_non_dom(arms_means[k], arms_means[j],0.);
        });
        deltas[k]=*std::max_element(tmp.begin(), tmp.end());
    }
    for(auto i: opt_arms){
        // d1
        std::transform(arms.begin(), arms.end(), tmp.begin(), [&arms_means, &i](size_t j){
            return minimum_quantity_dom(arms_means[i], arms_means[j], 0.) + INF*(i==j);
        });
        d1 = *std::min_element(tmp.begin(), tmp.end());
        //d2
        std::transform(arms.begin(), arms.end(), tmp.begin(), [&arms_means, &pareto_set_mask, &i](size_t j){
            return minimum_quantity_dom(arms_means[j], arms_means[i], 0.) + INF*((i==j)|| (!pareto_set_mask[j]));
        });
        d2 = *std::min_element(tmp.begin(), tmp.end());
        //d3
        std::transform(arms.begin(), arms.end(), tmp.begin(), [&](size_t j){
            return (std::max(minimum_quantity_dom(arms_means[j], arms_means[i],0.),0.) + deltas[j]) + INF*((i==j)||pareto_set_mask[j]);
        });
        d3 = *std::min_element(tmp.begin(), tmp.end());
        deltas[i] = std::min(std::min(d1, d2), d3);
    }
    return deltas;
}double get_z1t(const std::vector<std::vector<double>> &means, const std::vector<size_t>& St, const std::vector<size_t>& Ts, double eps_1, double delta, double sigma) {
    if (St.empty()) return INF; // should raise exception
    return std::transform_reduce(St.begin(), St.end(), INF,
                                 [](double xmin, double x){return std::min(xmin,x);},
                                 [&](size_t i){
                                     return get_h(i, means, Ts,eps_1,delta, sigma);
                                 });
}
double get_z2t(const std::vector<std::vector<double>> & means, const std::vector<size_t> & St_comp,
               const std::vector<size_t>& Ts, double& eps_1, double delta, double sigma) {
    if (St_comp.empty()) return INF;
    return std::transform_reduce(St_comp.begin(), St_comp.end(), INF,
                                 [](double xmin, double x) { return std::min(xmin, x); },
                                 [&](size_t i) {
                                     return std::max(get_h(i, means, Ts,eps_1, delta, sigma), get_g(i, means, Ts, std::vector<size_t>({}), 0, delta, sigma));
                                 });
}
