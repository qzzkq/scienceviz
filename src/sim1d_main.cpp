#include "core/sim1d.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: scienceviz_sim1d config.cfg\n";
        return 1;
    }
    try {
        svcore::Sim1D sim(svcore::load_config_1d(argv[1]));
        sim.run();
        std::cout << "time=" << sim.time() << " steps=" << sim.steps()
                  << " mass=" << sim.mass() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
