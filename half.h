/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Defines a class for using IEEE half-precision floating-point types in host or
      device code.
*/
#pragma once

#include <cmath>
#include <limits>
#include <cstdint>
#include <hip/hip_fp16.h>

#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

/// IEEE half-precision floating-point type
struct alignas(2) half_t {

  //
  // Data members
  //

  /// Storage type
  uint16_t storage;

  //
  // Static conversion operators
  //

  /// Constructs from an unsigned short
  CUTLASS_HOST_DEVICE
  static half_t bitcast(uint16_t x) {
    half_t h;
    h.storage = x;
    return h;
  }

  /// FP32 -> FP16 conversion - rounds to nearest even
  CUTLASS_HOST_DEVICE
  static half_t convert(float const& flt) {
    // software implementation rounds toward nearest even
    unsigned const& s = reinterpret_cast<unsigned const &>(flt);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      return bitcast(sign);
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      return bitcast(u);
    }

    int sticky_bit = 0;

    if (exp >= -14) {
      // normal fp32 to normal fp16
      exp = uint16_t(exp + uint16_t(15));
      u = uint16_t(((exp & 0x1f) << 10));
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);

        sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    // round to nearest even
    int round_bit = ((mantissa >> 12) & 1);
    sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
      u = uint16_t(u + 1);
    }

    u |= sign;

    return bitcast(u);
  }

  /// FP32 -> FP16 conversion - rounds to nearest even
  CUTLASS_HOST_DEVICE
  static half_t convert(int const& n) {
    return convert(float(n));
  }

  /// FP32 -> FP16 conversion - rounds to nearest even
  CUTLASS_HOST_DEVICE
  static half_t convert(unsigned const& n) {
    return convert(float(n));
  }

  /// Converts a half-precision value stored as a uint16_t to a float
  CUTLASS_HOST_DEVICE
  static float convert(half_t const& x) {
    uint16_t const &h = x.storage;
    int sign = ((h >> 15) & 1);
    int exp = ((h >> 10) & 0x1f);
    int mantissa = (h & 0x3ff);
    unsigned f = 0;

    if (exp > 0 && exp < 31) {
      // normal
      exp += 112;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    } else if (exp == 0) {
      if (mantissa) {
        // subnormal
        exp += 113;
        while ((mantissa & (1 << 10)) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= 0x3ff;
        f = (sign << 31) | (exp << 23) | (mantissa << 13);
      } else {
        // sign-preserving zero
        f = (sign << 31);
      }
    } else if (exp == 31) {
      if (mantissa) {
        f = 0x7fffffff;  // not a number
      } else {
        f = (0xff << 23) | (sign << 31);  //  inf
      }
    }
    return reinterpret_cast<float const&>(f);
  }

  //
  // Methods
  //

  /// Default constructor
  CUTLASS_HOST_DEVICE
  half_t() : storage(0) { }

  /// Reinterpret cast from CUDA's half type
  CUTLASS_HOST_DEVICE
  explicit half_t(half const & x): storage(reinterpret_cast<uint16_t const &>(x)) {

  }

  /// Floating point conversion
  CUTLASS_HOST_DEVICE
  explicit half_t(float x) {
    storage = convert(x).storage;
  }

  /// Floating point conversion
  CUTLASS_HOST_DEVICE
  explicit half_t(double x): half_t(float(x)) {

  }

  /// Integer conversion - round to nearest even
  CUTLASS_HOST_DEVICE
  explicit half_t(int x) {
    storage = convert(x).storage;
  }

  /// Integer conversion - round toward zero
  CUTLASS_HOST_DEVICE
  explicit half_t(unsigned x) {
    storage = convert(x).storage;
  }

  /// Assignment
  CUTLASS_HOST_DEVICE
  half_t & operator=(half const &x) {
    storage = reinterpret_cast<uint16_t const &>(x);
    return *this;
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  operator float() const {
    return convert(*this);
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  operator double() const {
    return double(convert(*this));
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(convert(*this));
  }

  /// Casts to bool
  CUTLASS_HOST_DEVICE
  operator bool() const {
    return (convert(*this) != 0.0f);
  }

  /// Bitcasts to CUDA's half type
  CUTLASS_HOST_DEVICE
  half to_half() const {
    return reinterpret_cast<half const &>(storage);
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  uint16_t& raw() {
    return storage;
  }

  /// Accesses raw internal state
  CUTLASS_HOST_DEVICE
  uint16_t raw() const {
    return storage;
  }

  /// Returns the sign bit
  CUTLASS_HOST_DEVICE
  bool signbit() const {
    return ((storage & 0x8000) != 0);
  }

  /// Returns the biased exponent
  CUTLASS_HOST_DEVICE
  int exponent_biased() const {
    return int((storage >> 10) & 0x1f);
  }

  /// Returns the unbiased exponent
  CUTLASS_HOST_DEVICE
  int exponent() const {
    return exponent_biased() - 15;
  }

  /// Returns the mantissa
  CUTLASS_HOST_DEVICE
  int mantissa() const {
    return int(storage & 0x3ff);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool signbit(half_t const& h) {
  return ((h.raw() & 0x8000) != 0);
}

CUTLASS_HOST_DEVICE
half_t abs(half_t const& h) {
  return half_t::bitcast(h.raw() & 0x7fff);
}

CUTLASS_HOST_DEVICE
bool isnan(half_t const& h) {
  return (h.exponent_biased() == 0x1f) && h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isfinite(half_t const& h) {
  return (h.exponent_biased() != 0x1f);
}

CUTLASS_HOST_DEVICE
half_t nanh(const char*) {
  // NVIDIA canonical NaN
  return half_t::bitcast(0x7fff);
}

CUTLASS_HOST_DEVICE
bool isinf(half_t const& h) {
  return (h.exponent_biased() == 0x1f) && !h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isnormal(half_t const& h) {
  return h.exponent_biased() && h.exponent_biased() != 0x1f;
}

CUTLASS_HOST_DEVICE
int fpclassify(half_t const& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x1f) {
    if (mantissa) {
      return FP_NAN;
    }
    else {
      return FP_INFINITE;
    }
  }
  else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    }
    else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

CUTLASS_HOST_DEVICE
half_t sqrt(half_t const& h) {
  return half_t(std::sqrt(float(h)));
}

CUTLASS_HOST_DEVICE
half_t copysign(half_t const& a, half_t const& b) {

  uint16_t a_mag = (reinterpret_cast<uint16_t const &>(a) & 0x7fff);
  uint16_t b_sign = (reinterpret_cast<uint16_t const &>(b) & 0x8000);
  uint16_t result = (a_mag | b_sign);

  return reinterpret_cast<half_t const &>(result);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool operator==(half_t const& lhs, half_t const& rhs) {
  return float(lhs) == float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator!=(half_t const& lhs, half_t const& rhs) {
  return float(lhs) != float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<(half_t const& lhs, half_t const& rhs) {
  return float(lhs) < float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<=(half_t const& lhs, half_t const& rhs) {
  return float(lhs) <= float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>(half_t const& lhs, half_t const& rhs) {
  return float(lhs) > float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>=(half_t const& lhs, half_t const& rhs) {
  return float(lhs) >= float(rhs);
}

CUTLASS_HOST_DEVICE
half_t operator+(half_t const& lhs, half_t const& rhs) {
  return half_t(float(lhs) + float(rhs));
}

CUTLASS_HOST_DEVICE
half_t operator-(half_t const& lhs) {
  return half_t(-float(lhs));
}

CUTLASS_HOST_DEVICE
half_t operator-(half_t const& lhs, half_t const& rhs) {
  return half_t(float(lhs) - float(rhs));
}

CUTLASS_HOST_DEVICE
half_t operator*(half_t const& lhs, half_t const& rhs) {
  return half_t(float(lhs) * float(rhs));
}

CUTLASS_HOST_DEVICE
half_t operator/(half_t const& lhs, half_t const& rhs) {
  return half_t(float(lhs) / float(rhs));
}

CUTLASS_HOST_DEVICE
half_t& operator+=(half_t & lhs, half_t const& rhs) {
  lhs = half_t(float(lhs) + float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator-=(half_t & lhs, half_t const& rhs) {
  lhs = half_t(float(lhs) - float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator*=(half_t & lhs, half_t const& rhs) {
  lhs = half_t(float(lhs) * float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator/=(half_t & lhs, half_t const& rhs) {
  lhs = half_t(float(lhs) / float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator++(half_t & lhs) {
  float tmp(lhs);
  ++tmp;
  lhs = half_t(tmp);
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t& operator--(half_t & lhs) {
  float tmp(lhs);
  --tmp;
  lhs = half_t(tmp);
  return lhs;
}

CUTLASS_HOST_DEVICE
half_t operator++(half_t & lhs, int) {
  half_t ret(lhs);
  float tmp(lhs);
  tmp++;
  lhs = half_t(tmp);
  return ret;
}

CUTLASS_HOST_DEVICE
half_t operator--(half_t & lhs, int) {
  half_t ret(lhs);
  float tmp(lhs);
  tmp--;
  lhs = half_t(tmp);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


//
// User-defined literals
//

CUTLASS_HOST_DEVICE
half_t operator "" _hf(long double x) {
  return half_t(float(x));
}

CUTLASS_HOST_DEVICE
half_t operator "" _hf(unsigned long long int x) {
  return half_t(int(x));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
