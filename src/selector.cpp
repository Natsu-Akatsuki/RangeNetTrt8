/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of rangenet_lib, and covered by the provided LICENSE file.
 *
 */

// selective network library (conditional build)
#include <selector.hpp>

// Only to be used with segmentation
namespace rangenet {
namespace segmentation {

/**
 * @brief Makes a network with the desired backend, checking that it exists,
 *        it is implemented, and that it was compiled.
 *
 * @return std::unique_ptr<Net>
 */
std::unique_ptr<Net> make_net(const std::string& path) {

  // make a network
  std::unique_ptr<Net> network = std::unique_ptr<Net>(new NetTensorRT(path));;
  return network;
}

}  // namespace segmentation
}  // namespace rangenet
