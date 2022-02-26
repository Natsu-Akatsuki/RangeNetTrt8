/*
 * copy from
 * https://stackoverflow.com/questions/43230542/write-vectorvectorfloat-to-binary-file
 */
#include "vector_io.h"

void saveVector(std::string path,
                const std::vector<std::vector<float>> &myVector) {
  std::ofstream FILE(path, std::ios::out | std::ofstream::binary);

  // Store size of the outer vector
  int s1 = myVector.size();
  FILE.write(reinterpret_cast<const char *>(&s1), sizeof(s1));

  // Now write each vector one by one
  for (auto &v : myVector) {
    // Store its size
    int size = v.size();
    FILE.write(reinterpret_cast<const char *>(&size), sizeof(size));

    // Store its contents
    FILE.write(reinterpret_cast<const char *>(&v[0]), v.size() * sizeof(float));
  }
  FILE.close();
}

void readVector(std::string path, std::vector<std::vector<float>> &myVector) {
  std::ifstream FILE(path, std::ios::in | std::ifstream::binary);

  int size = 0;
  FILE.read(reinterpret_cast<char *>(&size), sizeof(size));
  myVector.resize(size);
  for (int n = 0; n < size; ++n) {
    int size2 = 0;
    FILE.read(reinterpret_cast<char *>(&size2), sizeof(size2));
    float f;
    for (int k = 0; k < size2; ++k) {
      FILE.read(reinterpret_cast<char *>(&f), sizeof(f));
      myVector[n].push_back(f);
    }
  }
}

int main() {
  std::vector<std::vector<float>> ff;
  ff.resize(10);
  ff[0].push_back(10);
  ff[0].push_back(12);
  saveVector("test.bin", ff);

  std::vector<std::vector<float>> ff2;
  readVector("test.bin", ff2);

  if (ff == ff2) {
    std::cout << "ok!";
  }
}