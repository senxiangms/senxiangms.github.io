//===- kernel_operator.h - the instructions, and the umbrella header ------===//
//
// Everything here is an *instruction on a pipe*, and that is the only thing
// that matters for correctness:
//
//   DataCopy(local, global)   MTE2   GM -> UB
//   DataCopy(global, local)   MTE3   UB -> GM
//   Add/Sub/Mul/Duplicate...  V      UB -> UB
//
// Each one ticks its pipe's clock, checks that whatever it touches has been
// handed over properly, and records what it wrote. Include this header and you
// have the whole mini Ascend C.
//
//===----------------------------------------------------------------------===//
#ifndef MINI_ASCENDC_KERNEL_OPERATOR_H
#define MINI_ASCENDC_KERNEL_OPERATOR_H

#include "kernel_common.h"
#include "kernel_tensor.h"
#include "kernel_tpipe.h"

namespace AscendC {

/// A vector instruction chews through 256 bytes per repeat; a count that does
/// not fill a repeat still costs a whole one. Only used for the trace, but it
/// is the number you tune tile sizes against.
constexpr uint32_t ONE_REPEAT_BYTE_SIZE = 256;

namespace detail {

inline bool CheckDmaAlign(const char *what, uint32_t bytes)
{
    if (bytes % ONE_BLK_SIZE == 0) { return true; }
    ReportSyncError("%s moves %u bytes, which is not a multiple of %u - DMA transfers whole "
                    "32B blocks and would clobber the tail", what, bytes, ONE_BLK_SIZE);
    return false;
}

template <typename T>
inline bool CheckFits(const char *what, const LocalTensor<T> &t, uint32_t count)
{
    if (count * sizeof(T) <= t.GetLength()) { return true; }
    ReportSyncError("%s: %u elements do not fit in a %u byte buffer", what, count, t.GetLength());
    return false;
}

/// Read side of a local tensor: check the hand-over, then remember the read so
/// that an unsynchronised overwrite is caught too.
template <typename T>
inline void TouchRead(const LocalTensor<T> &t, uint32_t count, PipeId pipe, const char *what,
                      uint64_t tick)
{
    Hardware hw = t.GetHardware();
    uint32_t off = t.GetBufferAddr();
    uint32_t bytes = count * sizeof(T);
    CheckAccess(hw, off, bytes, pipe, what);
    RecordRead(hw, off, bytes, pipe, tick);
}

template <typename T>
inline void TouchWrite(const LocalTensor<T> &t, uint32_t count, PipeId pipe, const char *what,
                       uint64_t tick)
{
    Hardware hw = t.GetHardware();
    uint32_t off = t.GetBufferAddr();
    uint32_t bytes = count * sizeof(T);
    CheckAccess(hw, off, bytes, pipe, what, /*isWrite=*/true);
    RecordWrite(hw, off, bytes, pipe, tick);
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// DataCopy - the MTE pipes
//===----------------------------------------------------------------------===//

/// GM -> local. Issued on MTE2.
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
                                uint32_t count)
{
    uint32_t bytes = count * sizeof(T);
    detail::CheckDmaAlign("DataCopy(GM->local)", bytes);
    detail::CheckFits("DataCopy(GM->local)", dstLocal, count);

    uint64_t tick = PipeTick(PipeId::MTE2);
    detail::TouchWrite(dstLocal, count, PipeId::MTE2, "DataCopy(GM->local)", tick);
    std::memcpy(dstLocal.GetPhyAddr(), srcGlobal.GetPhyAddr(), bytes);
    Core().bytesIn += bytes;
    Trace(PipeId::MTE2, "DataCopy gm -> %s[0x%05x] %u B", ToString(dstLocal.GetHardware()),
          dstLocal.GetBufferAddr(), bytes);
}

/// local -> GM. Issued on MTE3.
template <typename T>
__aicore__ inline void DataCopy(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
                                uint32_t count)
{
    uint32_t bytes = count * sizeof(T);
    detail::CheckDmaAlign("DataCopy(local->GM)", bytes);
    detail::CheckFits("DataCopy(local->GM)", srcLocal, count);

    uint64_t tick = PipeTick(PipeId::MTE3);
    detail::TouchRead(srcLocal, count, PipeId::MTE3, "DataCopy(local->GM)", tick);
    std::memcpy(const_cast<T *>(dstGlobal.GetPhyAddr()), srcLocal.GetPhyAddr(), bytes);
    Core().bytesOut += bytes;
    Trace(PipeId::MTE3, "DataCopy %s[0x%05x] -> gm %u B", ToString(srcLocal.GetHardware()),
          srcLocal.GetBufferAddr(), bytes);
}

//===----------------------------------------------------------------------===//
// Vector instructions - the V pipe
//===----------------------------------------------------------------------===//

#define MINI_DEFINE_VEC_BINARY(NAME, EXPR)                                                       \
    template <typename T>                                                                        \
    __aicore__ inline void NAME(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,  \
                                const LocalTensor<T> &src1Local, int32_t calCount)                \
    {                                                                                            \
        uint32_t n = static_cast<uint32_t>(calCount);                                            \
        detail::CheckFits(#NAME, dstLocal, n);                                                   \
        uint64_t tick = PipeTick(PipeId::V);                                                     \
        detail::TouchRead(src0Local, n, PipeId::V, #NAME, tick);                                 \
        detail::TouchRead(src1Local, n, PipeId::V, #NAME, tick);                                 \
        detail::TouchWrite(dstLocal, n, PipeId::V, #NAME, tick);                                 \
        T *dst = dstLocal.GetPhyAddr();                                                          \
        const T *a = src0Local.GetPhyAddr();                                                     \
        const T *b = src1Local.GetPhyAddr();                                                     \
        for (uint32_t i = 0; i < n; ++i) { dst[i] = EXPR; }                                      \
        Trace(PipeId::V, "%s %s[0x%05x] = f(0x%05x, 0x%05x) %u elems (%u repeats)", #NAME,       \
              ToString(dstLocal.GetHardware()), dstLocal.GetBufferAddr(),                        \
              src0Local.GetBufferAddr(), src1Local.GetBufferAddr(), n,                           \
              (n * (unsigned)sizeof(T) + ONE_REPEAT_BYTE_SIZE - 1) / ONE_REPEAT_BYTE_SIZE);      \
    }

MINI_DEFINE_VEC_BINARY(Add, a[i] + b[i])
MINI_DEFINE_VEC_BINARY(Sub, a[i] - b[i])
MINI_DEFINE_VEC_BINARY(Mul, a[i] * b[i])
MINI_DEFINE_VEC_BINARY(Div, a[i] / b[i])
MINI_DEFINE_VEC_BINARY(Max, a[i] > b[i] ? a[i] : b[i])
MINI_DEFINE_VEC_BINARY(Min, a[i] < b[i] ? a[i] : b[i])

#undef MINI_DEFINE_VEC_BINARY

#define MINI_DEFINE_VEC_SCALAR(NAME, EXPR)                                                       \
    template <typename T>                                                                        \
    __aicore__ inline void NAME(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,   \
                                T scalarValue, int32_t calCount)                                  \
    {                                                                                            \
        uint32_t n = static_cast<uint32_t>(calCount);                                            \
        detail::CheckFits(#NAME, dstLocal, n);                                                   \
        uint64_t tick = PipeTick(PipeId::V);                                                     \
        detail::TouchRead(srcLocal, n, PipeId::V, #NAME, tick);                                  \
        detail::TouchWrite(dstLocal, n, PipeId::V, #NAME, tick);                                 \
        T *dst = dstLocal.GetPhyAddr();                                                          \
        const T *a = srcLocal.GetPhyAddr();                                                      \
        const T s = scalarValue;                                                                 \
        for (uint32_t i = 0; i < n; ++i) { dst[i] = EXPR; }                                      \
        Trace(PipeId::V, "%s %s[0x%05x] %u elems", #NAME, ToString(dstLocal.GetHardware()),      \
              dstLocal.GetBufferAddr(), n);                                                      \
    }

MINI_DEFINE_VEC_SCALAR(Adds, a[i] + s)
MINI_DEFINE_VEC_SCALAR(Muls, a[i] * s)

#undef MINI_DEFINE_VEC_SCALAR

/// Fill a local tensor with a constant. Also on the V pipe - which is why a
/// Duplicate right after a DataCopy into the same buffer still needs a flag.
template <typename T>
__aicore__ inline void Duplicate(const LocalTensor<T> &dstLocal, T scalarValue, int32_t calCount)
{
    uint32_t n = static_cast<uint32_t>(calCount);
    detail::CheckFits("Duplicate", dstLocal, n);
    uint64_t tick = PipeTick(PipeId::V);
    detail::TouchWrite(dstLocal, n, PipeId::V, "Duplicate", tick);
    T *dst = dstLocal.GetPhyAddr();
    for (uint32_t i = 0; i < n; ++i) { dst[i] = scalarValue; }
    Trace(PipeId::V, "Duplicate %s[0x%05x] %u elems", ToString(dstLocal.GetHardware()),
          dstLocal.GetBufferAddr(), n);
}

}  // namespace AscendC

#endif  // MINI_ASCENDC_KERNEL_OPERATOR_H
