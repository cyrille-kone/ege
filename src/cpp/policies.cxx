#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include<functional>
#include "utils.hpp"
#include "bandits.hpp"
#include "policies.hpp"
#include <exception>
policy_fb::policy_fb(bandit &bandit_ref) {
    this->bandit_ref = &bandit_ref;
    K = bandit_ref.K;
    D = bandit_ref.D;
    dim = bandit_ref.D;
    sigma = bandit_ref.sigma;
    action_space = bandit_ref.action_space;
}
std::pair<std::pair<size_t, bool>, std::vector<size_t>> policy_fb::loop(size_t, size_t) {
    return {};
}

ua::ua(bandit &bandit_ref) : policy_fb(bandit_ref){};

bool ua::loop(const size_t seed, const size_t T) {
    bandit_ref->reset_env(seed);
    std::vector<std::vector<double>> total_rewards(K, std::vector<double>(D, 0.));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    std::vector<bool> Sr_mask(K);
    std::vector<std::vector<double>> batch_pulls;
    size_t num_pulls = T / K;
    std::vector<size_t> Nc(K, num_pulls);
    for(size_t a: action_space){
        batch_pulls = std::move(bandit_ref->sample(std::move(std::vector<size_t>(num_pulls, a))));

        for(size_t t{0}; t<num_pulls; ++t){
            for(size_t d=0; d<D; ++d) total_rewards[a][d] += batch_pulls[t][d];
        }
        for (size_t d{0}; d < D; ++d) {
            means[a][d] = total_rewards[a][d] / (double) Nc[a];
        }
    }
    size_t remaining = T - num_pulls*K;
    for (size_t i{0}; i < remaining; ++i) {
        batch_pulls = std::move(bandit_ref->sample({i}));
        for (size_t d {0}; d < D; ++d) {
            total_rewards[i][d] += batch_pulls[0][d];
        }
        Nc[i] += 1;
    }
    for(size_t a:action_space){
        for (size_t d{0}; d < D; ++d) {
            means[a][d] = total_rewards[a][d] / (double) Nc[a];
        }
    }
    Sr_mask = pareto_optimal_arms_mask(means);
    return std::equal(Sr_mask.begin(), Sr_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
}
ege_sh::ege_sh(bandit &bandit_ref):policy_fb(bandit_ref){};

bool ege_sh::loop(const size_t seed, const size_t T) {
    bandit_ref->reset_env(seed);
    std::vector<bool> active_mask(K, true);
    std::vector<size_t> Nc(K, 0);
    std::vector<bool> accept_mask(K, false);
    std::vector<std::vector<double>> total_rewards(K, std::vector<double>(D, 0.));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    std::vector<size_t> temp_vec{action_space};
    std::vector<bool> Sr_mask(K);
    size_t ceil_log_K  = std::ceil(std::log2(K));
    size_t Nr, Nr_keep, Nr_remove; // number of active arms, number of arms to keep and to remove
    std::vector<std::vector<double>> batch_pulls;
    std::vector<double> vec_sub_gap(K);
    std::vector<double> vec_delta_star(K);
    double threshold;
    size_t count_remove, counter, a_, num_pulls;
    for (size_t r{0}; r < ceil_log_K; ++r) {
        //std::cout<<num_pulls<<" "<<r<<std::endl;
        Nr = std::accumulate(active_mask.begin(), active_mask.end(), size_t{0});
        num_pulls = std::floor(T/((double)(Nr*ceil_log_K)));
        if (num_pulls>0){
            for (size_t i{0}; i < K; ++i) {
                if (active_mask[i]){
                    // pull active arms
                    batch_pulls = bandit_ref->sample(std::vector<size_t>(num_pulls, i));
                    for(size_t t{0}; t<num_pulls; ++t){
                        for(size_t d{0}; d<D; ++d) total_rewards[i][d] += batch_pulls[t][d];
                    }
                    Nc[i] += num_pulls;
                    for (size_t d{0}; d < D; ++d) {
                        means[i][d] = total_rewards[i][d] / (double) Nc[i];
                    }
                }
            }
        }
        Sr_mask = pareto_optimal_arms_mask(means, active_mask);
        vec_delta_star = delta_star(means, active_mask);
        for(size_t a: action_space){
            if (active_mask[a]){
                vec_sub_gap[a] = sub_opt_gap(a, means, vec_delta_star, active_mask, Sr_mask[a]) ;//+1e-4*(!Sr_mask[a]);
            }
        }
        Nr_keep = std::ceil((double)Nr/ 2.);
        Nr_remove = Nr - Nr_keep;
        // tie-breaking rule
        std::sort(temp_vec.begin(), temp_vec.end(), [&vec_sub_gap, &active_mask](size_t i, size_t j){
            return vec_sub_gap[i] - INF*(!active_mask[i]) < vec_sub_gap[j] - INF*(!active_mask[j]);
        });
        // any arm with a gap larger than this threshold will be discarded
        threshold = vec_sub_gap[temp_vec[(K-Nr) + Nr_keep]];
        try
        {
            std::sort(temp_vec.begin()+(K-Nr), temp_vec.end(),  [&Sr_mask, &vec_sub_gap](size_t i, size_t j){
                double x{vec_sub_gap[i] - (!Sr_mask[i])*INF}, y{vec_sub_gap[j] -(!Sr_mask[j])*INF};
                if (x==y) return i<j;
                return  x<y;});
        }
        catch (std::exception& e)
        {
            std::cout << "Standard exception: " << e.what() << std::endl;
        }
        count_remove = 0;
        counter = 0;
        while(count_remove< Nr_remove){
            a_ = temp_vec[counter];
           if( active_mask[a_] && vec_sub_gap[a_]>= threshold){
               accept_mask[a_] = Sr_mask[a_];
               active_mask[a_] = false;
               ++count_remove;
           }
           ++counter;
        }
    }
    accept_mask = std::move(sum(accept_mask, active_mask));
    return std::equal(accept_mask.begin(), accept_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
}




ege_sr::ege_sr(bandit &bandit_ref):policy_fb(bandit_ref){};
ege_srk::ege_srk(bandit &bandit_ref):policy_fb(bandit_ref){};


bool ege_sr::loop(const size_t seed, const size_t T, const size_t k ) {
    bandit_ref->reset_env(seed);
    std::vector<size_t> n_ks(K);
    std::vector<bool> active_mask(K, true);
    std::vector<size_t> Nc(K, 0);
    std::vector<bool> accept_mask(K, false);
    std::vector<std::vector<double>> total_rewards(K, std::vector<double>(D, 0.));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    std::vector<bool> Sr_mask(K);
    // std::vector<size_t> to_pull;
    std::vector<std::vector<double>> batch_pulls;
    std::vector<double> vec_sub_gap(K);
    std::vector<double> vec_delta_star(K);
    size_t opt_arms_found{0};
    size_t i_r;
    double log_K = std::accumulate(action_space.begin(), action_space.end(), -1./2, []( double acc, size_t i ){return acc + 1./(double)(i+1); });
    double I_1, I_2;
    size_t a_r, d_r;
    size_t num_pulls ;
    // check early stopping
    bool flag_es {false};
    n_ks[0] = 0;
    for (size_t r = 1; r < K; ++r) {
        n_ks[r] = std::ceil((1./log_K)* (T-K) /(K+1. - r));
    }

    for (size_t r{1}; r < K; ++r) {
        num_pulls = n_ks[r] - n_ks[r-1];
        if (num_pulls>0){
            for (size_t i {0}; i < K; ++i) {
                if (active_mask[i]){
                    // pull active arms
                    batch_pulls = std::move(bandit_ref->sample(std::move(std::vector<size_t>(num_pulls, i))));
                    for(size_t t{0}; t<num_pulls; ++t){
                        for(size_t d{0}; d<D; ++d) total_rewards[i][d] += batch_pulls[t][d];
                    }
                    Nc[i] += num_pulls;
                    // actualize the empirical means
                    for (size_t d{0}; d < D; ++d) {
                        means[i][d] = total_rewards[i][d] / (double) Nc[i];
                    }

                }
            }
        }
        Sr_mask = std::move(pareto_optimal_arms_mask(means, active_mask));
        vec_delta_star = std::move(delta_star(means, active_mask));
        I_1 = -INF;
        I_2 = -INF;
        for (size_t a:action_space) {
            if (active_mask[a]){
                vec_sub_gap[a] = sub_opt_gap(a, means, vec_delta_star, active_mask, Sr_mask[a]);
                switch (Sr_mask[a]) {
                    case true:
                        // optimal
                        if (vec_sub_gap[a]>I_1) {
                            a_r = a; I_1 = vec_sub_gap[a];
                        }
                        break;
                    case false: // sub-optimal
                        if (vec_sub_gap[a]>I_2) {
                            d_r = a; I_2 = vec_sub_gap[a];
                        }
                        break;
                }
            }
        }
        if (I_2>=I_1){
            // reject sub-optimal arm
            i_r = d_r;
            accept_mask[i_r] = false;
        }
        else {
            i_r = a_r;
            accept_mask[i_r] = true;
        }
        // deactivate the removed arm
        active_mask[i_r] = false;
        if (std::accumulate(accept_mask.begin(), accept_mask.end(), 0)>= k){ flag_es = true; break;}
    }
    if (! flag_es) accept_mask = std::move(sum(accept_mask, active_mask));
    for(auto a:action_space){
        opt_arms_found += (accept_mask[a] && accept_mask[a]==bandit_ref->optimal_arms_mask[a]);
    }
    return flag_es? opt_arms_found ==k: std::equal(accept_mask.begin(), accept_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
}

std::pair<bool, std::pair<std::vector<size_t>, std::pair<size_t, size_t>>> ege_srk::loop(const size_t seed, const size_t T, const size_t k ) {
    bandit_ref->reset_env(seed);
    std::vector<size_t> n_ks(K);
    std::vector<bool> active_mask(K, true);
    std::vector<size_t> Nc(K, 0);
    std::vector<bool> accept_mask(K, false);
    std::vector<std::vector<double>> total_rewards(K, std::vector<double>(D, 0.));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    std::vector<bool> Sr_mask(K);
    std::vector<std::vector<double>> batch_pulls;
    std::vector<double> vec_sub_gap(K);
    std::vector<double> vec_delta_star(K);
    size_t opt_arms_found{0};
    size_t i_r;
    double log_K = std::accumulate(action_space.begin(), action_space.end(), -1./2, []( double acc, size_t i ){return acc + 1./(double)(i+1); });
    double I_1, I_2;
    size_t a_r, d_r;
    size_t num_pulls ;
    // check early stopping
    bool flag_es {false};
    n_ks[0] = 0;
    for (size_t r = 1; r < K; ++r) {
        n_ks[r] = std::ceil((1./log_K)* (T-K) /(K+1. - r));
    }
    size_t r;
    for (r = 1; r < K; ++r) {
        num_pulls = n_ks[r] - n_ks[r-1];
        if (num_pulls>0){
            for (size_t i {0}; i < K; ++i) {
                if (active_mask[i]){
                    // pull active arms
                    batch_pulls = std::move(bandit_ref->sample(std::move(std::vector<size_t>(num_pulls, i))));
                    for(size_t t{0}; t<num_pulls; ++t){
                        for(size_t d{0}; d<D; ++d) total_rewards[i][d] += batch_pulls[t][d];
                    }
                    Nc[i] += num_pulls;
                    // actualize the empirical means
                    for (size_t d{0}; d < D; ++d) {
                        means[i][d] = total_rewards[i][d] / (double) Nc[i];
                    }
                }
            }
        }
        Sr_mask = std::move(pareto_optimal_arms_mask(means, active_mask));
        vec_delta_star = std::move(delta_star(means, active_mask));
        I_1 = -INF;
        I_2 = -INF;
        for (size_t a:action_space) {
            if (active_mask[a]){
                vec_sub_gap[a] = sub_opt_gap(a, means, vec_delta_star, active_mask, Sr_mask[a]);
                switch (Sr_mask[a]) {
                    case true:
                        // optimal
                        if (vec_sub_gap[a]>I_1) {
                            a_r = a; I_1 = vec_sub_gap[a];
                        }
                        break;
                    case false: // sub-optimal
                        if (vec_sub_gap[a]>I_2) {
                            d_r = a; I_2 = vec_sub_gap[a];
                        }
                        break;
                }
            }
        }
        if (I_2>=I_1){
            // reject sub-optimal arm
            i_r = d_r;
            accept_mask[i_r] = false;
        }
        else {
            i_r = a_r;
            accept_mask[i_r] = true;
        }
        // deactivate the removed arm
        active_mask[i_r] = false;
        if (std::accumulate(accept_mask.begin(), accept_mask.end(), 0)>= k){ flag_es = true; break;}
    }
    if (! flag_es) accept_mask = std::move(sum(accept_mask, active_mask));
    std::vector<size_t> found_set;
    found_set.reserve(K);
    for(auto a:action_space){
        opt_arms_found += (accept_mask[a] && accept_mask[a]==bandit_ref->optimal_arms_mask[a]);
        if (accept_mask[a]) found_set.push_back(a);
    }
    bool found = flag_es? opt_arms_found ==k: std::equal(accept_mask.begin(), accept_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end());
    size_t N_tau = std::accumulate(Nc.begin(), Nc.end(), size_t{0});
    std::pair<size_t, size_t> tau_N_tau = {r, N_tau};
    return  std::pair<bool, std::pair<std::vector<size_t>, std::pair<size_t, size_t>>>{found, std::pair<std::vector<size_t>, std::pair<size_t, size_t>>{found_set, tau_N_tau}};// optimal_set found
}

std::vector<double> batch_sr(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds,size_t k ){
    std::vector<double> ans(Ts.size());
    //thread_local
    size_t N{seeds.size()};
    ege_sr sr(bandit_ref);
    size_t count{0};
    size_t i, j;
    for(i=0;i<Ts.size(); ++i){
        count = 0;
#pragma omp parallel for default(none)  shared(Ts, seeds, i, k) firstprivate(sr)  reduction(+:count)
        for(j=0; j<seeds.size(); ++j){
            count += (size_t)sr.loop(seeds[j], Ts[i], k);
        }
//#pragma omp barrier
        ans[i] = (double)count / (double) N;
    }
    return ans;
}

std::pair<std::vector<double>, std::pair<std::vector<std::vector<std::vector<size_t>>>, std::pair<std::vector<double>, std::vector<double>>>> batch_srk(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds,size_t k ){
    std::vector<double> ans(Ts.size());
    std::vector<double> taus(Ts.size());
    std::vector<double> N_taus(Ts.size());
    std::vector<std::vector<std::vector<size_t>>> found_sets(Ts.size(), std::vector<std::vector<size_t>>(seeds.size()));
    //thread_local
    size_t N{seeds.size()};
    ege_srk srk(bandit_ref);
    size_t count{0}, tau{0}, N_tau{0};
    size_t i, j;
    for(i=0;i<Ts.size(); ++i){
        count = 0;
        N_tau = 0;
        tau = 0;
#pragma omp parallel for default(none)  shared(Ts, seeds, i, k, found_sets) firstprivate(srk)  reduction(+:count, tau, N_tau)
        for(j=0; j<seeds.size(); ++j){
            auto res = srk.loop(seeds[j], Ts[i], k);
            count += (size_t)res.first;
            tau += res.second.second.first;
            N_tau += res.second.second.second;
            found_sets[i][j] = res.second.first;
        }
//#pragma omp barrier
        ans[i] = (double)count / (double) N;
        taus[i] = (double)tau / (double) N;
        N_taus[i] = (double)N_tau / (double) N;
    }
    return std::pair<std::vector<double>, std::pair<std::vector<std::vector<std::vector<size_t>>>, std::pair<std::vector<double>, std::vector<double>>>>{
        ans, std::pair<std::vector<std::vector<std::vector<size_t>>>, std::pair<std::vector<double>, std::vector<double>>>{found_sets, std::pair<std::vector<double>, std::vector<double>>{taus, N_taus}}
    };
}

std::vector<double> batch_ua(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds){
    std::vector<double> ans(Ts.size());
    //thread_local
    size_t N{seeds.size()};
    ua u_a(bandit_ref);
    size_t i;
    size_t j;
    size_t count;
    for(i=0;i<Ts.size(); ++i){
        count = 0;
#pragma omp parallel  for private(j) firstprivate(u_a) default(none) shared(seeds, Ts, i) reduction(+:count)
        for(j=0; j<seeds.size(); ++j){
            count +=  u_a.loop(seeds[j], Ts[i]);
        }
//#pragma omp barrier
        ans[i] = (double)count / (double) N;
    }
    return ans;
}
std::vector<double> batch_sh(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds){
    size_t count;
    std::vector<double> ans(Ts.size());
    //thread_local
    size_t N{seeds.size()};
    ege_sh sh(bandit_ref);
    size_t i;
    size_t j;
    for(i=0;i<Ts.size(); ++i){
        count = 0;
#pragma omp parallel for private(j) firstprivate(sh) default(none) shared(seeds, Ts, i) reduction(+:count)
        for(j=0; j<seeds.size(); ++j){
            count +=  sh.loop(seeds[j], Ts[i]);
        }
        ans[i] = (double) count / (double) N;
    }
    return ans;
}