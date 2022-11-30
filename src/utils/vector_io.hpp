#ifndef RANGENET_LIB_VECTOR_IO_H
#define RANGENET_LIB_VECTOR_IO_H
#include <fstream>
#include <iostream>

#include <string>
#include <vector>
void saveVector(std::string path,
                const std::vector<std::vector<float>> &myVector);
void readVector(std::string path, std::vector<std::vector<float>> &myVector);

#endif // RANGENET_LIB_VECTOR_IO_H
