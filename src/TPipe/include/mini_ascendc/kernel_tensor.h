//===- kernel_tensor.h - GlobalTensor and LocalTensor ---------------------===//
//
// Two views on memory, and the difference between them is the whole point:
//
//   GlobalTensor<T>  a typed pointer into GM (HBM). Big, slow, addressed with
//                    a 64-bit offset. Nothing computes on it directly.
//   LocalTensor<T>   a typed window into a *core-local* memory, described by
//                    (logical position, byte offset, length) - not by a host
//                    pointer, because UB is only 192KB and is addressed by
//                    offset. It also remembers which TQue buffer it came from,
//                    which is how FreeTensor(t) can find its way back.
//
//===----------------------------------------------------------------------===//
#ifndef MINI_ASCENDC_KERNEL_TENSOR_H
#define MINI_ASCENDC_KERNEL_TENSOR_H

#include "kernel_common.h"

namespace AscendC {

//===----------------------------------------------------------------------===//
// Buffer bookkeeping shared with TQue
//===----------------------------------------------------------------------===//

/// The lifecycle of one TQue buffer. Every transition is checked; the state
/// machine is what turns a misuse of the queue into a message instead of
/// silently corrupted data.
enum class TBufState : uint8_t {
    FREE,      // owned by nobody
    OCCUPIED,  // handed out by AllocTensor, being filled
    ENQUE,     // published by EnQue, waiting for a consumer
    DEQUE,     // taken by DeQue, being consumed
};

inline const char *ToString(TBufState s)
{
    switch (s) {
        case TBufState::FREE:     return "FREE";
        case TBufState::OCCUPIED: return "OCCUPIED";
        case TBufState::ENQUE:    return "ENQUE";
        default:                  return "DEQUE";
    }
}

/// One buffer slot inside a queue. `depth` of these per TQue.
struct TBufType {
    TBufState state = TBufState::FREE;
    Hardware hard = Hardware::UB;
    TPosition logicPos = TPosition::VECIN;
    uint32_t address = 0;   // byte offset inside its physical memory
    uint32_t dataLen = 0;   // byte length
    TEventID enQueEvtID = INVALID_TEVENTID;    // raised by EnQue, waited by DeQue
    TEventID freeBufEvtID = INVALID_TEVENTID;  // raised by FreeTensor, waited by AllocTensor
};

using TBufHandle = TBufType *;

/// What a LocalTensor actually stores about where it points.
struct TBuffAddr {
    TPosition logicPos = TPosition::VECIN;
    TBufHandle bufferHandle = nullptr;
    uint32_t bufferAddr = 0;
    uint32_t dataLen = 0;
};

//===----------------------------------------------------------------------===//
// LocalTensor
//===----------------------------------------------------------------------===//

template <typename T>
class LocalTensor {
public:
    using PrimType = T;

    __aicore__ inline LocalTensor() {}

    __aicore__ inline explicit LocalTensor(const TBuffAddr &addr) : address_(addr) {}

    __aicore__ inline TPosition GetPosition() const { return address_.logicPos; }
    __aicore__ inline Hardware GetHardware() const { return GetPhyType(address_.logicPos); }

    /// Length in elements (Ascend C's GetSize; GetLength is the byte count).
    __aicore__ inline uint32_t GetSize() const { return address_.dataLen / sizeof(T); }
    __aicore__ inline uint32_t GetLength() const { return address_.dataLen; }
    __aicore__ inline void SetSize(uint32_t size) { address_.dataLen = size * sizeof(T); }

    __aicore__ inline uint32_t GetBufferAddr() const { return address_.bufferAddr; }
    __aicore__ inline TBufHandle GetBufferHandle() const { return address_.bufferHandle; }
    __aicore__ inline const TBuffAddr &GetAddr() const { return address_; }

    /// Raw pointer into the simulated local memory. On hardware this is a UB
    /// offset, not something you can dereference from the host.
    __aicore__ inline T *GetPhyAddr(uint32_t offset = 0) const
    {
        return reinterpret_cast<T *>(Core().Base(GetHardware()) + address_.bufferAddr) + offset;
    }

    /// Scalar read/write go through the S pipe - they need flags against V and
    /// MTE just like anything else, and that is what most "my tensor is zero"
    /// bugs turn out to be.
    __aicore__ inline T GetValue(uint32_t index) const
    {
        uint64_t tick = PipeTick(PipeId::S);
        uint32_t byteOff = address_.bufferAddr + index * sizeof(T);
        CheckAccess(GetHardware(), byteOff, sizeof(T), PipeId::S, "GetValue");
        RecordRead(GetHardware(), byteOff, sizeof(T), PipeId::S, tick);
        return *GetPhyAddr(index);
    }

    __aicore__ inline void SetValue(uint32_t index, T value) const
    {
        uint64_t tick = PipeTick(PipeId::S);
        Hardware hw = GetHardware();
        uint32_t byteOff = address_.bufferAddr + index * sizeof(T);
        CheckAccess(hw, byteOff, sizeof(T), PipeId::S, "SetValue", /*isWrite=*/true);
        *GetPhyAddr(index) = value;
        RecordWrite(hw, byteOff, sizeof(T), PipeId::S, tick);
    }

    __aicore__ inline T &operator()(uint32_t index) const { return *GetPhyAddr(index); }

    /// A sub-window starting `offset` elements in. It keeps the same buffer
    /// handle, so FreeTensor on a slice still frees the right slot.
    __aicore__ inline LocalTensor<T> operator[](uint32_t offset) const
    {
        TBuffAddr addr = address_;
        addr.bufferAddr += offset * sizeof(T);
        addr.dataLen -= offset * sizeof(T);
        return LocalTensor<T>(addr);
    }

    template <typename U>
    __aicore__ inline LocalTensor<U> ReinterpretCast() const
    {
        return LocalTensor<U>(address_);
    }

    inline void Print(uint32_t len = 8) const
    {
        uint32_t n = len < GetSize() ? len : GetSize();
        std::printf("  ub[0x%05x] %u elems:", address_.bufferAddr, GetSize());
        for (uint32_t i = 0; i < n; ++i) { std::printf(" %g", (double)*GetPhyAddr(i)); }
        std::printf("%s\n", n < GetSize() ? " ..." : "");
    }

private:
    TBuffAddr address_;
};

//===----------------------------------------------------------------------===//
// GlobalTensor
//===----------------------------------------------------------------------===//

template <typename T>
class GlobalTensor {
public:
    using PrimType = T;

    __aicore__ inline GlobalTensor() {}

    __aicore__ inline void SetGlobalBuffer(__gm__ T *buffer, uint64_t bufferSize)
    {
        address_ = buffer;
        size_ = bufferSize;
    }

    __aicore__ inline void SetGlobalBuffer(__gm__ T *buffer) { address_ = buffer; }

    __aicore__ inline const __gm__ T *GetPhyAddr() const { return address_; }
    __aicore__ inline __gm__ T *GetPhyAddr(uint64_t offset) const { return address_ + offset; }
    __aicore__ inline uint64_t GetSize() const { return size_; }

    __aicore__ inline T GetValue(uint64_t offset) const
    {
        PipeTick(PipeId::S);
        return address_[offset];
    }

    __aicore__ inline void SetValue(uint64_t offset, T value)
    {
        PipeTick(PipeId::S);
        address_[offset] = value;
    }

    __aicore__ inline __gm__ T &operator()(uint64_t offset) const { return address_[offset]; }

    __aicore__ inline GlobalTensor<T> operator[](uint64_t offset) const
    {
        GlobalTensor<T> ret;
        ret.SetGlobalBuffer(address_ + offset, size_ > offset ? size_ - offset : 0);
        return ret;
    }

private:
    __gm__ T *address_ = nullptr;
    uint64_t size_ = 0;
};

}  // namespace AscendC

#endif  // MINI_ASCENDC_KERNEL_TENSOR_H
