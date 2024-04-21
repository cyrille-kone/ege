#include <vector>
#include <cstdlib>
#include <iostream>
//#include<omp.h>
#include "src/cpp/utils.hpp"
#include "src/cpp/bandits.hpp"
#include "src/cpp/policies.hpp"
#include <chrono>
int main(int argc, char** argv) {
    std::vector<std::vector<double>> data{{0.75785738, 0.04866446},
                                          {0.4445091 , 0.19905503},
                                          {0.45422223, 0.9413393 },
                                          {0.89876041, 0.05671574},
                                          {0.80762673, 0.22822175},};
    /*bernoulli bern(data);
    gaussian gauss(data, {.5, .5});
    ege_sr sr(bern);
    psi_ape_fb ape_fb(bern);
    ege_sh sh(bern);
    psi_ucbe ucbe(gauss);
    psi_ucbe_adapt ucbe_a(bern);*/
    std::vector<size_t> seeds(5000);
    std::iota(seeds.begin(), seeds.end(), size_t{0});

    double result{0.};
    size_t i;
    bernoulli bern(data);
    ege_sr sr(bern);
    ege_sh sh(bern);

/*
#pragma omp parallel for num_threads(8) reduction(+:result)
 */
    for(i=0;i<500; ++i){
        // cout<<auer.loop(i, 0.1, 0).first<<endl;
        // auto res = ape_fb.loop(i, 500, 0.2);
        auto res = sh.loop(i, 500);
        //std::cout<<"[seed="<<i<<"]: "<<std::boolalpha<<res<<" thread_id="<<omp_get_thread_num()<<"\n";
        // auto ret_val = bandit.sample({0,1,2});
        // print_array2d(ret_val);
        result += res;
    }
    std::cout<<result/500<<std::endl;
    std::cout<<bern.H<<std::endl;
    return 0;
}