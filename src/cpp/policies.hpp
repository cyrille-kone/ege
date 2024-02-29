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

struct psi_ape_fb: policy_fb{
    psi_ape_fb()= default;
    explicit psi_ape_fb(bandit&);
    [[nodiscard]] bool loop(const size_t&, const size_t&, const double&);
    size_t get_ct(const std::vector<std::vector<double>> & means, std::size_t bt, const std::vector<std::vector<double>>& beta) {
        std::vector<double> tmp(K);
        // out les transform reduce inutiles
        std::transform(action_space.begin(), action_space.end(), tmp.begin(),[&](size_t i){
            return minimum_quantity_dom(means[bt], means[i], 0) - beta[bt][i] + INF*(i==bt);
        });
        return get_argmin(tmp, action_space);}
    size_t get_bt(const std::vector<std::vector<double>>& means, const std::vector<bool>& opt_mask, const std::vector<std::vector<double>>& beta){
        std::vector<double> res;
        res.reserve(K);
        double res_;
        std::transform(action_space.begin(), action_space.end(), std::back_inserter(res), [&](size_t i){
            if (opt_mask[i]) return -INF;
            res_ = INF;
            for(size_t j:action_space){
                res_ = std::min(res_, M(means[i], means[j]) + beta[i][j] + INF*(i==j));
            }
            return res_;
        });
        return (action_space)[std::distance(res.begin(), std::max_element(res.begin(), res.end()))];
        //return res;

    }
};

struct psi_ucbe: policy_fb{
    psi_ucbe()= default;
    explicit psi_ucbe(bandit&);
    [[nodiscard]] bool loop(const size_t&, const size_t& T, const double& a);
    inline double beta(const size_t& t, const double a) const{return  sqrt(a/ (double)t);};
    size_t get_yt(const std::vector<std::vector<double>> &means, const size_t& ht, const std::vector<double>& beta_vec);
    size_t get_st(const std::vector<std::vector<double>>&, size_t& , const std::vector<double>&) const;
    std::pair<double, size_t> get_z2l_t(const std::vector<std::vector<double>> &means,
                                                  const std::vector<bool> &St_mask, const std::vector<double> &beta_vec);
    std::pair<double, size_t> get_z1h_t(const std::vector<std::vector<double>> &means, const std::vector<bool>&St_mask, const std::vector<double>& beta_vec);
};

struct psi_ucbe_adapt: psi_ucbe{
    psi_ucbe_adapt()= default;
    explicit psi_ucbe_adapt(bandit&);
    [[nodiscard]] bool loop(const size_t&, const size_t& T, const double& c);
};

struct ape_b: policy_fb{
    ape_b()= default;
    explicit ape_b(bandit&);
    [[nodiscard]] bool loop(const size_t&, const size_t&, const double&);
    inline double beta(const size_t& t, const double a) const{return (2./5)*sqrt(a/ (double)t);};
    inline size_t get_ct(const std::vector<std::vector<double>> & means, std::size_t bt, const std::vector<double>& beta_vec) {
        size_t res{bt};
        double min_{INF}, tmp;
        for (size_t a: action_space) {
          tmp =   M(means[bt], means[a]) - beta_vec[bt]-beta_vec[a] + INF*(a==bt);
          if (tmp <min_){
              res = a; min_ = tmp;
          }}
        return res;};

    inline size_t get_bt(const std::vector<std::vector<double>>& means, const std::vector<bool>& opt_mask, const std::vector<double>& beta_vec){
        //std::vector<double> res;
        size_t bt;
        double res_, min_{INF};
        //res.reserve(K);
        for(auto opt: opt_mask){
           if (!opt) goto Opt_non_full;
       }
        // if Opt is full
        res_ = INF;
        for(size_t i:action_space){
            min_ = INF;
            for(size_t j:action_space){
                min_ = std::min(min_, M(means[i], means[j]) - beta_vec[i]- beta_vec[j] + INF*(i==j));
            }
            if (min_<res_){
                res_ = min_;
                bt = i;
            }
        }
        return bt;
        /*goto Opt_full;
        Opt_full:

        std::transform(action_space.begin(), action_space.end(), std::back_inserter(res), [&](size_t i){
            if (opt_mask[i]) return -INF;
            res_ = INF;
            for(size_t j:action_space){
                res_ = std::min(res_, M(means[i], means[j]) - beta_vec[i]- beta_vec[j] + INF*(i==j));
            }
            return res_;
        });
        return (action_space)[std::distance(res.begin(), std::min_element(res.begin(), res.end()))];
*/
        Opt_non_full:
        res_ = -INF;
        for(size_t i:action_space){
            if(opt_mask[i]) continue;
            min_ = INF;
            for(size_t j:action_space){
                min_ = std::min(min_, M(means[i], means[j]) + beta_vec[i]+ beta_vec[j] + INF*(i==j));
            }
            if (min_>res_){
                res_ = min_;
                bt = i;
            }
        }
        return bt;
  /*
          std::transform(action_space.begin(), action_space.end(), std::back_inserter(res), [&](size_t i){
            if (opt_mask[i]) return -INF;
            res_ = INF;
            for(size_t j:action_space){
                res_ = std::min(res_, M(means[i], means[j]) + beta_vec[i]+ beta_vec[j] + INF*(i==j));
            }
            return res_;
        });
          return (action_space)[std::distance(res.begin(), std::max_element(res.begin(), res.end()))];*/
    }
};


struct ape_b_adapt: ape_b{
    ape_b_adapt()= default;
    explicit ape_b_adapt(bandit&);
    [[nodiscard]] bool loop(const size_t&, const size_t& T, const double& c);
};


std::vector<double> batch_sr(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds,size_t k );
std::pair<std::vector<double>, std::pair<std::vector<std::vector<std::vector<size_t>>>, std::pair<std::vector<double>, std::vector<double>>>> batch_srk(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds,size_t k );
std::vector<double> batch_ua(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds);
std::vector<double> batch_sh(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds);
std::vector<double> batch_ape(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds, double c);
std::vector<double> batch_ape_adapt(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds, double c);