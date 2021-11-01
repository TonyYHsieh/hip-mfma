#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

// "Constant" address space for kernel parameter
#define CONSTANT __attribute__((address_space(4)))

namespace ck {

// index type
using index_t = int32_t;

} // namespace ck
#endif
