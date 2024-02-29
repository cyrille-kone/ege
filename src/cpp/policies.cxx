#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include<functional>
#include "utils.hpp"
#include "bandits.hpp"
#include "policies.hpp"
#include <exception>

// TODO compute St wisely
// TODO verifier tout le code
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
    // std::vector<size_t> to_pull;
    std::vector<std::vector<double>> batch_pulls;
    // check early stopping
    size_t num_pulls = T / K;
    std::vector<size_t> Nc(K, num_pulls);
    for(size_t a: action_space){
        // pull active arms

        batch_pulls = std::move(bandit_ref->sample(std::move(std::vector<size_t>(num_pulls, a))));

        for(size_t t{0}; t<num_pulls; ++t){
            for(size_t d=0; d<D; ++d) total_rewards[a][d] += batch_pulls[t][d];
        }
        // actualize the empirical means
        //std::cout<<Nc[i]<<std::endl;
        for (size_t d{0}; d < D; ++d) {
            means[a][d] = total_rewards[a][d] / (double) Nc[a];
        }
        //  std::cout<<" inside loop"<<std::endl;
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

        // print_array2d(means);
        Sr_mask = pareto_optimal_arms_mask(means, active_mask);
        // print_array1d(active_mask);
        // print_array1d(Sr_mask);
        vec_delta_star = delta_star(means, active_mask);
        for(size_t a: action_space){
            if (active_mask[a]){
                vec_sub_gap[a] = sub_opt_gap(a, means, vec_delta_star, active_mask, Sr_mask[a]) ;//+1e-4*(!Sr_mask[a]);
            }
        }
        //std::cout<<" cout gaps"<<std::endl;
        //print_array1d(vec_delta_star);
        //print_array1d(vec_sub_gap);

        Nr_keep = std::ceil((double)Nr/ 2.);
        Nr_remove = Nr - Nr_keep;
        //std::cout<<Nr_remove<<std::endl;
        // tie-breaking rule
        std::sort(temp_vec.begin(), temp_vec.end(), [&vec_sub_gap, &active_mask](size_t i, size_t j){
            return vec_sub_gap[i] - INF*(!active_mask[i]) < vec_sub_gap[j] - INF*(!active_mask[j]);
        });
        //std::cout<<"sort ok"<<std::endl;
        //print_array1d(temp_vec);

        // any arm with a gap larger than this threshold will be discarded
        threshold = vec_sub_gap[temp_vec[(K-Nr) + Nr_keep]];
        //print_array1d(temp_vec);
        //std::cout<<threshold<<std::endl;
        //print_array1d(vec_sub_gap);
        // sort by optimality
        //@note the
        //std::cout<<" threshok ok "<< threshold <<std::endl;
        //std::cout<<"debugging"<<std::endl;
        //std::cout<<K-Nr<<std::endl;
        //,
        //, [&Sr_mask](size_t i, size_t j){
        //            return !Sr_mask[i];}
        //print_array1d(Sr_mask);
        //print_array1d(active_mask);
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

        //std::cout<<" new sort ok "<<std::endl;
        //return false;
        //print_array1d(temp_vec);
        //return false;

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
    //print_array1d(bandit_ref->optimal_arms);
    //print_array1d(accept_mask);
    //std::cout<<std::accumulate(Nc.begin(), Nc.end(), 0)<<std::endl;
    // std::cout<<"final"<<std::endl;
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
        //std::cout<<log_K<<std::endl;
        //std::cout<<n_ks[r]<<std::endl;
    }

    for (size_t r{1}; r < K; ++r) {
        //std::cout<<num_pulls<<std::endl;
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
                    //std::cout<<Nc[i]<<std::endl;
                    for (size_t d{0}; d < D; ++d) {
                        means[i][d] = total_rewards[i][d] / (double) Nc[i];
                    }
                  //  std::cout<<" inside loop"<<std::endl;
                }
            }
        }
        // print_array2d(means);
        Sr_mask = std::move(pareto_optimal_arms_mask(means, active_mask));
        // print_array1d(active_mask);
        // print_array1d(Sr_mask);
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
        //std::cout<<I_1<<std::endl;
        //std::cout<<I_2<<std::endl;
        //print_array1d(vec_sub_gap);
        //print_array1d(active_mask);
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
    //print_array1d(bandit_ref->optimal_arms);
    //print_array1d(accept_mask);
    //std::cout<<std::accumulate(Nc.begin(), Nc.end(), 0)<<std::endl;
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
        //std::cout<<log_K<<std::endl;
        //std::cout<<n_ks[r]<<std::endl;
    }
    size_t r;
    for (r = 1; r < K; ++r) {
        //std::cout<<num_pulls<<std::endl;
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
                    //std::cout<<Nc[i]<<std::endl;
                    for (size_t d{0}; d < D; ++d) {
                        means[i][d] = total_rewards[i][d] / (double) Nc[i];
                    }
                    //  std::cout<<" inside loop"<<std::endl;
                }
            }
        }
        // print_array2d(means);
        Sr_mask = std::move(pareto_optimal_arms_mask(means, active_mask));
        // print_array1d(active_mask);
        // print_array1d(Sr_mask);
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
        //std::cout<<I_1<<std::endl;
        //std::cout<<I_2<<std::endl;
        //print_array1d(vec_sub_gap);
        //print_array1d(active_mask);
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
    //print_array1d(bandit_ref->optimal_arms);
    //print_array1d(accept_mask);
    //std::cout<<std::accumulate(Nc.begin(), Nc.end(), 0)<<std::endl;
    std::vector<size_t> found_set;
    found_set.reserve(K);
    for(auto a:action_space){
        opt_arms_found += (accept_mask[a] && accept_mask[a]==bandit_ref->optimal_arms_mask[a]);
        if (accept_mask[a]) found_set.push_back(a);
    }
    bool found = flag_es? opt_arms_found ==k: std::equal(accept_mask.begin(), accept_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end());
    size_t N_tau = std::accumulate(Nc.begin(), Nc.end(), size_t{0});
    //std::cout<<r<< " "<<N_tau<<std::endl;
    std::pair<size_t, size_t> tau_N_tau = {r, N_tau};
    return  std::pair<bool, std::pair<std::vector<size_t>, std::pair<size_t, size_t>>>{found, std::pair<std::vector<size_t>, std::pair<size_t, size_t>>{found_set, tau_N_tau}};// optimal_set found
}

psi_ape_fb::psi_ape_fb(bandit &bandit_ref): policy_fb(bandit_ref) {
};

bool psi_ape_fb::loop(const size_t& seed, const size_t& T, const double& delta) {
    double cg = Cg(delta);
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<bool> St_mask;
    std::vector<bool> opt_mask(K);
    std::vector<size_t> Ts(K, 1);
    std::vector<std::vector<double>> beta (K, std::vector<double>(K, betaij(1, 1, cg, sigma)));
    std::vector<std::vector<double>> means_t(K, std::vector<double>(D));
// compute the empirical Pareto set St
#define get_St2 {St_mask  = std::move(pareto_optimal_arms_mask(means_t));}; \
// compute OPT(t)
#define get_opt2 { for(size_t a: action_space){                                                                           \
       opt_mask[a] =   get_h(a, means_t, beta) > 0;                                                                                                  \
    }  };                                                                                                     \
    size_t at, bt, ct;
// Initial sampling
    for (auto k:action_space){
        means_t[k] = bandit_ref->sample({k})[0];
    }
    get_St2
    get_opt2
    size_t t = K;
    while(t<T){
        bt = get_bt(means_t, opt_mask, beta);
        ct = get_ct(means_t, bt, beta);
        at = (Ts[bt]>Ts[ct])?ct:bt;
        for (auto k: {at}) {
            std::vector<double> v(std::move(bandit_ref->sample({k})[0])); // to move
            std::transform(means_t[k].begin(), means_t[k].end(), v.begin(), means_t[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
        }
        // update beta
        for (size_t i = 0; i < K; ++i) {
            beta[at][i] = betaij(Ts[i], Ts[at],cg, sigma);
            beta[i][at] = beta[at][i];
        }
        get_St2
        get_opt2
        ++t;
    }
    // optimal_arms is always sorted;
    St_mask = pareto_optimal_arms_mask(means_t);
    // print_array2d(beta);
    // std::cout<<std::accumulate(Ts.begin(), Ts.end(), 0)<<std::endl;
    // print_array1d(Ts);
    //  std::cout<<cg<<std::endl;
    return std::equal(St_mask.begin(), St_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
}





psi_ucbe::psi_ucbe(bandit &bandit_ref): policy_fb(bandit_ref){};

std::pair<double, size_t> psi_ucbe::get_z1h_t(const std::vector<std::vector<double>> &means, const std::vector<bool>& St_mask, const std::vector<double>& beta_vec) {
    //useless
    // if (St.empty()) return INF;
    double min_{INF}, r_;
    size_t a_min_{1<<12};
    for (size_t a: action_space) {
        if (!St_mask[a]) continue;
        r_ = get_h(a, means, beta_vec);
        if (r_ < min_) {
            a_min_ = a;
            min_ = r_;
        }
    }
    return std::pair<double, size_t>{min_, a_min_};
};

std::pair<double, size_t> psi_ucbe::get_z2l_t(const std::vector<std::vector<double>> &means,
                                              const std::vector<bool> &St_mask, const std::vector<double> &beta_vec) {
    // remplacer par un assert
    //if (St_comp.empty()) return INF;
    double min_{INF}, r_;
    size_t a_min_{MAX_K};
    for( auto a: action_space) {
        if (St_mask[a]) continue;
        r_ = get_g(a, means, beta_vec);
        if(r_< min_){
            min_ = r_;
            a_min_ = a;
        }
    };
    return std::pair<double, int>{min_, a_min_};
}


size_t psi_ucbe::get_yt(const std::vector<std::vector<double>> &means, const size_t& ht, const std::vector<double>& beta_vec) {

    return std::accumulate(action_space.begin(), action_space.end(), ht,
                           [&ht, &means, &beta_vec](size_t amin, size_t k){
                               return ((M(means[ht], means[amin]) - beta_vec[amin] +INF*(amin==ht)) < (M(means[ht], means[k]) - beta_vec[k] + INF*(k==ht)))?amin:k;
                           });
}

size_t psi_ucbe::get_st(const std::vector<std::vector<double>> &means, size_t &lt, const std::vector<double> &beta_vec) const {
    return std::accumulate(action_space.begin(), action_space.end(), lt,
                           [&lt, &means, &beta_vec](size_t km, size_t k){
                               return ((m(means[lt], means[km]) + beta_vec[km] -INF*(km==lt)) < (m(means[lt], means[k]) + beta_vec[k] - INF*(k==lt)))?k:km;
                           });
}
bool psi_ucbe::loop(const size_t& seed, const size_t& T, const double& a) {
    // Initialize the model
    bandit_ref->reset_env(seed);
    //auto res = bandit_ref->sample({0,1});
    //print_array2d(res);
    // std::vector<size_t> At(4); // to contain ht yt lt st
    std::vector<size_t> At;
    std::vector<bool> St_mask;
    std::vector<size_t> Ts(std::vector<size_t>(K, 1));
    std::vector<double> betas(K, beta(1, a));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    size_t ht, yt, lt, st;
    std::pair<double, size_t> z_a;

    for (auto k:action_space){
        means[k] = bandit_ref->sample({k})[0];
    }
    St_mask = pareto_optimal_arms_mask(means);
    size_t t = K;
    double z1_t, z2_t;
    size_t c = 0;
    size_t max_K{MAX_K};
    while((t+4)<=T){
        z_a = get_z1h_t(means, St_mask, betas);
        z1_t = z_a.first;
        ht = z_a.second;
        z_a = get_z2l_t(means, St_mask, betas);
        z2_t = z_a.first;
        lt = z_a.second;
        yt = get_yt(means, ht, betas);
        if(lt==max_K) At = {ht, yt};
        else {
            // only here st is computed
            st = get_st(means, lt,  betas);
            At = {ht, yt, lt, st};
        };
        //std::cout<<ht<<" "<<yt<<" "<<lt<< " "<<st<<std::endl;
        // if(c==7) break;
        //At = {ht, yt, lt, st};
        // print_array1d(At);
        //std::cout<<"Inside function "<<c<<std::endl;
        //std::cout<<ht<<" "<<yt<<" "<<lt<< " "<<st<<std::endl;
        //std::cout<<At.size()<<std::endl;
        //for (auto k: At){
        //    std::cout<<k<<std::endl;
        //}
        // break;
        for (auto k:At) {
            std::vector<double> v = std::move(bandit_ref->sample({k})[0]); // to move
            std::transform(means[k].begin(), means[k].end(), v.begin(), means[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
            betas[k] = beta(Ts[k], a);
        }

        t+= At.size();
        St_mask = std::move(pareto_optimal_arms_mask(means));

        //print_array1d(St_mask);
        //++c;
        if (c==-1){
            std::cout<<"special debugging"<<std::endl;
            print_array1d(St_mask);
            z_a = get_z1h_t(means, St_mask, betas);
            z1_t = z_a.first;
            ht = z_a.second;
            z_a = get_z2l_t(means, St_mask, betas);
            z2_t = z_a.first;
            lt = z_a.second;
            yt = get_yt(means, ht, betas);
            st = get_st(means, lt,  betas);
            std::cout<<ht<<" "<<yt<<" "<<lt<< " "<<st<<std::endl;
            if(lt==max_K) At = {ht, yt};
            else At = {ht, yt,lt, st};
            std::cout<<At.size()<<std::endl;
            for (auto k: At){
                std::cout<<k<<std::endl;
            }
            break;
        }
    }
    // optimal_arms is always sorted;
    return std::equal(St_mask.begin(), St_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
};
psi_ucbe_adapt::psi_ucbe_adapt(bandit &bandit_ref): psi_ucbe(bandit_ref){};
bool psi_ucbe_adapt::loop(const size_t &seed, const size_t &T, const double &c) {
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<size_t> At;
    std::vector<double> vec_delta_star(K);
    std::vector<bool> active_mask(K, true);
    std::vector<double> vec_sub_gaps(K);
    std::vector<bool> St_mask;
    std::vector<size_t> Ts(std::vector<size_t>(K, 1));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    size_t ht, yt, lt, st;
    std::pair<double, size_t> z_a;
    double H_emp;
    for (auto k:action_space){
        means[k] = bandit_ref->sample({k})[0];
    }
    St_mask = pareto_optimal_arms_mask(means);
    vec_delta_star = std::move(delta_star(means,active_mask));
    for (auto k: action_space) {
        vec_sub_gaps[k] = sub_opt_gap(k, means, vec_delta_star, active_mask, St_mask[k]);
    }
    H_emp = std::accumulate(vec_sub_gaps.begin(), vec_sub_gaps.end(), 0., [](double acc, double x){
        return acc + 1./std::pow((x+1e-8), 2);
    });
    std::vector<double> betas(K, beta(1, (double)T*c/H_emp));
    size_t t = K;
    double z1_t, z2_t;
    size_t max_K{MAX_K};
    while((t+4)<=T){
        z_a = get_z1h_t(means, St_mask, betas);
        z1_t = z_a.first;
        ht = z_a.second;
        z_a = get_z2l_t(means, St_mask, betas);
        z2_t = z_a.first;
        lt = z_a.second;
        yt = get_yt(means, ht, betas);
        if(lt==max_K) At = {ht, yt};
        else {
            // only here st is computed
            st = get_st(means, lt,  betas);
            At = {ht, yt, lt, st};
        };
        for (auto k:At) {
            std::vector<double> v = std::move(bandit_ref->sample({k})[0]); // to move
            std::transform(means[k].begin(), means[k].end(), v.begin(), means[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
            betas[k] = beta(Ts[k], c*(double)(T-K)/H_emp);
        }

        t+= At.size();
        St_mask = std::move(pareto_optimal_arms_mask(means));
        vec_delta_star = std::move(delta_star(means,active_mask));
        for (auto k: action_space) {
            vec_sub_gaps[k] = sub_opt_gap(k, means, vec_delta_star, active_mask, St_mask[k]);
        }
        H_emp = std::accumulate(vec_sub_gaps.begin(), vec_sub_gaps.end(), 0., [](double acc, double x){
            return acc + 1./std::pow((x+1e-8), 2);
        });
    }
    return std::equal(St_mask.begin(), St_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found

}
ape_b::ape_b(bandit &bandit_ref): policy_fb(bandit_ref){};

bool ape_b::loop(const size_t &seed, const size_t &T, const double & a) {
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<bool> St_mask;
    std::vector<bool> opt_mask(K);
    std::vector<size_t> Ts(K, 1);
    std::vector<double> beta_vec (K, beta(1, a));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    //std::vector<std::vector<double>> total_rewards(K, std::vector<double>(D, 0.));
    std::vector<std::vector<double>> batch_pull;
    std::vector<size_t> At;
    size_t at, bt, ct;
// compute the empirical Pareto set St
#define get_St {St_mask  = std::move(pareto_optimal_arms_mask(means));}; \
// compute OPT(t)
#define get_Opt { for(size_t k: action_space){                                                                           \
opt_mask[k] =   get_h(k, means, beta_vec) > 0;                                                                                                  \
    }  };                                                                                                     \
// Initial sampling
    for (size_t k:action_space){
        means[k] = bandit_ref->sample({k})[0];
    }
    //get_St
    get_Opt
    size_t t = K;
    while(t<T){
        bt = get_bt(means, opt_mask, beta_vec);
        ct = get_ct(means, bt, beta_vec);
        at = (Ts[bt]>Ts[ct])?ct:bt;
        // we sample both bt and ct to be faster
        At = {bt, ct};
        batch_pull = bandit_ref->sample(At); // to move
        //print_array1d(At);
        for (size_t i{0}; i < batch_pull.size(); ++i) {
            for (size_t d{0}; d < D; ++d) {
                means[At[i]][d] = ((means[At[i]][d] *(double)Ts[At[i]]) + batch_pull[i][d])/ ((double) Ts[At[i]] +1.);
            }
            ++Ts[At[i]];
            beta_vec[At[i]] = beta(Ts[At[i]], a);
        }
        //get_St
        get_Opt
        t+=2;
    }
    St_mask = pareto_optimal_arms_mask(means);
    return std::equal(St_mask.begin(), St_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
};

ape_b_adapt::ape_b_adapt(bandit &bandit_ref): ape_b(bandit_ref){};

bool ape_b_adapt::loop(const size_t &seed, const size_t &T, const double &c) {
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<double> vec_delta_star(K);
    std::vector<bool> active_mask(K, true);
    std::vector<double> beta_vec(K);
    std::vector<double> vec_sub_gaps(K);
    std::vector<bool> St_mask;
    std::vector<size_t> Ts(std::vector<size_t>(K, 1));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    double H_emp;
    size_t at, bt, ct;
    std::vector<bool> opt_mask(K);

#define get_H_emp St_mask = pareto_optimal_arms_mask(means); \
vec_delta_star = std::move(delta_star(means,active_mask)); \
    for (auto k: action_space) {\
        vec_sub_gaps[k] = sub_opt_gap(k, means, vec_delta_star, active_mask, St_mask[k]);\
    } \
    H_emp = std::accumulate(vec_sub_gaps.begin(), vec_sub_gaps.end(), 0., [](double acc, double x){\
        return acc + 1./std::pow((x+1e-8), 2); \
    });

    // compute the empirical Pareto set St
#define get_St_adapt {St_mask  = std::move(pareto_optimal_arms_mask(means));}; \
                                                                         \
// compute OPT(t)
#define get_Opt_adapt { for(size_t k: action_space){                                                                           \
opt_mask[k] =   get_h(k, means, beta_vec) > 0;                                                                                                  \
    }  };                                                                                                     \
// Initial sampling
    for (auto k:action_space){
        means[k] = bandit_ref->sample({k})[0];
    }
    get_St_adapt
    get_Opt_adapt
    size_t t = K;
    while(t+1<T){
        bt = get_bt(means, opt_mask, beta_vec);
        ct = get_ct(means, bt, beta_vec);
        at = (Ts[bt]>Ts[ct])?ct:bt;
        for (auto k: {bt, ct}) {
            std::vector<double> v(std::move(bandit_ref->sample({k})[0])); // to move
            std::transform(means[k].begin(), means[k].end(), v.begin(), means[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
        }
        // update beta
        beta_vec[at] = beta(Ts[at], (c*(25./36.)*(double)(T-K))/H_emp);
        get_St_adapt
        get_Opt_adapt
        get_H_emp
        t+=2;
    }
    // optimal_arms is always sorted;
    St_mask = pareto_optimal_arms_mask(means);
    return std::equal(St_mask.begin(), St_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
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

std::vector<double> batch_ape(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds, double c){
    std::vector<double> ans(Ts.size());
    //std::vector<std::vector<bool>> ret_val(Ts.size(), std::vector<bool>(seeds.size()));
    size_t N{seeds.size()};
    ape_b ape(bandit_ref);
    size_t count{0};
    size_t i;
    size_t j;
    double H{bandit_ref.H};
    size_t K{bandit_ref.K};
// #pragma omp parallel for private(i)  default(none) shared(ans, c, seeds, Ts, K, H, N) reduction(+:count) firstprivate(ape)
    for(i=0;i<Ts.size(); ++i){
        count = 0;
#pragma omp parallel for private(j) default(none) shared(c, i,  seeds, Ts, K, H) reduction(+:count) firstprivate(ape)
        for(j=0; j<seeds.size(); ++j){
            //ape_b ape(bandit_ref);
            count+= ape.loop(seeds[j], Ts[i], c*(25./36.)*(Ts[i]-K)/H);
        }
//#pragma omp barrier
//ans[i] = (double)std::accumulate(ret_val[i].begin(), ret_val[i].end(), size_t{0}) / (double) N;
ans[i] = count / (double)N;
    }
    return ans;
}
std::vector<double> batch_ape_adapt(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds, double c){
    std::vector<double> ans(Ts.size());
    size_t N{seeds.size()};
    ape_b_adapt ape_adapt(bandit_ref);
    size_t count;
    size_t i, j;
    for(i=0;i<Ts.size(); ++i){
        count = 0;
#pragma omp parallel for private(j) firstprivate(ape_adapt) default(none) shared(c, i, seeds, Ts) reduction(+:count)
        for(j=0; j<seeds.size(); ++j){
            count +=  ape_adapt.loop(seeds[j], Ts[i], c);
        }
//#pragma omp barrier
        ans[i] = ((double) count) / (double) N;
    }
    return ans;
}