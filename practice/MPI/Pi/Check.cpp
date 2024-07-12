#include <fstream>
#include <iostream>
#include <iomanip>

#define err 1e-15

int main() {
    std::ifstream ans("pi.bin", std::ios::binary | std::ios::in);
    std::ifstream check("ans.bin", std::ios::binary | std::ios::in);

    if(!ans.is_open() || !check.is_open()) {
        std::cerr << "Error: file not found" << std::endl;
        return 1;
    }

    long double ans_pi, check_pi;
    ans.read(reinterpret_cast<char*>(&ans_pi), sizeof(ans_pi));
    check.read(reinterpret_cast<char*>(&check_pi), sizeof(check_pi));
    if (std::abs(ans_pi - check_pi) < err) {
        std::cout << "Correct!" << std::endl;
    } else {
        std::cout << "Incorrect!" << std::endl;
        std::cout << std::fixed << std::setprecision(18) << "ans = " << ans_pi << std::endl;
        std::cout << std::fixed << std::setprecision(18) << "yours = " << check_pi << std::endl;
    }
    return 0;
}