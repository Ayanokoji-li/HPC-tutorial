#include <fstream>
#include <iostream>
#include <string>

#define DATA_PATH "./attachment/"
#define EPS 1e-6
#define FILE_NUM 4

int main(int argc, char const *argv[])
{
    int file_num = 4;
    if(argc == 2)
    {
        file_num = std::stoi(argv[1]);
    }
    for(int i = 1; i <= file_num; i ++)
    {
        std::ifstream f1((std::string(DATA_PATH) + std::to_string(i) + ".ref").c_str(), std::ios::binary);
        std::ifstream f2((std::string(DATA_PATH) + std::to_string(i) + ".ans").c_str(), std::ios::binary);
        if(!f1.is_open() || !f2.is_open())
        {
            std::cout << "File " << std::to_string(i) << " not found" << std::endl;
            continue;
        }
        uint64_t n1, n2, n3;
        f1.read((char *)&n1, 8);
        f1.read((char *)&n3, 8);
        double *c1 = (double *)malloc(n1 * n3 * 8);
        double *c2 = (double *)malloc(n1 * n3 * 8);
        f1.read((char *)c1, n1 * n3 * 8);
        f2.read((char *)c2, n1 * n3 * 8);

        bool flag = true;
        for(int j = 0; j < n1 * n3; j++)
        {
            if(abs(c1[j] - c2[j]) > EPS)
            {
                std::cout << "Test " << i << " failed" << std::endl;
                flag = false;
                break;
            }
        }
        if(flag)
        {
            std::cout << "Test " << i << " passed" << std::endl;
        }
    }    
}