//===- kernel_tpipe.h - TPipe, TQue, TBuf --------------------------------===//
//
// TPipe owns the core's local memories; TQue owns a slice of one of them plus
// the synchronisation that makes it safe to pass across pipes.
//
// The four queue calls map one-to-one onto four flag operations:
//
//   AllocTensor   WaitFlag<freeBufEvt>   "consumer is done with this slot"
//   EnQue         SetFlag <enQueEvt>     "producer has filled it"
//   DeQue         WaitFlag<enQueEvt>     "...so the consumer may read it"
//   FreeTensor    SetFlag <freeBufEvt>   "consumer is done, recycle it"
//
// With depth == 1 those four calls degenerate into a lock step. With depth == 2
// the producer can be filling slot B while the consumer still reads slot A -
// double buffering, and nothing in the kernel source had to change except the
// number in the template argument.
//
//===----------------------------------------------------------------------===//
#ifndef MINI_ASCENDC_KERNEL_TPIPE_H
#define MINI_ASCENDC_KERNEL_TPIPE_H

#include "kernel_common.h"
#include "kernel_tensor.h"

namespace AscendC {

/// Total number of buffer slots one TPipe can hand out, across all queues.
constexpr uint8_t QBUF_MAX_LEN = 64;

class TPipe;
template <TPosition pos>
class TBuf;

inline TPipe *&TPipePtrRef();
inline TPipe *GetTPipePtr();

//===----------------------------------------------------------------------===//
// TPipe
//===----------------------------------------------------------------------===//

/// One per kernel, usually the first local variable in it. TPipe is a bump
/// allocator over each physical memory plus a pool of event ids - there is no
/// free list, because a kernel's buffer layout is decided once and then reused
/// for every tile.
class TPipe {
public:
    __aicore__ inline TPipe() { Init(); }
    __aicore__ inline ~TPipe() { Destroy(); }

    __aicore__ inline void Init()
    {
        for (uint8_t h = 0; h < static_cast<uint8_t>(Hardware::MAX); ++h) { bufPool_[h].maxAddr = 0; }
        for (uint8_t e = 0; e < EVENT_NUM; ++e) { eventOccupy_[e] = 0; }
        curBufSize_ = 0;
        destroyed_ = false;
        TPipePtrRef() = this;
    }

    /// Carve `num` buffers of `len` bytes each out of the memory that `que`
    /// lives in. This is the only place local memory is ever handed out.
    template <class T>
    __aicore__ inline bool InitBuffer(T &que, uint8_t num, uint32_t len)
    {
        constexpr Hardware hard = T::bufHardType;
        constexpr TPosition logicPos = T::position;
        if (len % ONE_BLK_SIZE != 0) {
            ReportSyncError("InitBuffer: len %u is not a multiple of %u bytes; every local "
                            "buffer must be 32B aligned", len, ONE_BLK_SIZE);
            return false;
        }
        if (num > T::queDepth) {
            ReportSyncError("InitBuffer: %u buffers requested but the queue is declared with "
                            "depth %d", (unsigned)num, (int)T::queDepth);
            return false;
        }
        if (curBufSize_ + num > QBUF_MAX_LEN) {
            ReportSyncError("InitBuffer: out of buffer descriptors (max %u)", (unsigned)QBUF_MAX_LEN);
            return false;
        }
        uint32_t start = bufPool_[static_cast<uint8_t>(hard)].maxAddr;
        uint64_t need = static_cast<uint64_t>(num) * len;
        if (start + need > HardwareSize(hard)) {
            ReportSyncError("InitBuffer: %s overflow - %u x %u B requested, 0x%x already in use, "
                            "capacity %u B", ToString(hard), (unsigned)num, len, start,
                            HardwareSize(hard));
            return false;
        }

        TBufType *ptr = buf_ + curBufSize_;
        for (uint8_t i = 0; i < num; ++i) {
            ptr[i].state = TBufState::FREE;
            ptr[i].hard = hard;
            ptr[i].logicPos = logicPos;
            ptr[i].address = start + i * len;
            ptr[i].dataLen = len;
            ptr[i].enQueEvtID = INVALID_TEVENTID;
            ptr[i].freeBufEvtID = INVALID_TEVENTID;
        }
        que.InitBufHandles(ptr, num);
        bufPool_[static_cast<uint8_t>(hard)].maxAddr = start + static_cast<uint32_t>(need);
        curBufSize_ += num;

        Trace(PipeId::S, "InitBuffer %s x%u x%uB -> %s[0x%05x, 0x%05x)", T::PositionName(),
              (unsigned)num, len, ToString(hard), start, start + (unsigned)need);
        return true;
    }

    /// A TBuf is a plain scratch allocation: one buffer, no queue, no events.
    template <TPosition pos>
    __aicore__ inline bool InitBuffer(TBuf<pos> &buf, uint32_t len);

    /// Grab one of the hardware's flag registers for this event type.
    template <HardEvent evt>
    __aicore__ inline TEventID AllocEventID()
    {
        uint16_t &occ = eventOccupy_[static_cast<uint8_t>(evt)];
        for (int32_t i = 0; i < EVENT_ID_NUM; ++i) {
            if ((occ & (1u << i)) == 0) {
                occ |= static_cast<uint16_t>(1u << i);
                return static_cast<TEventID>(i);
            }
        }
        ReportSyncError("out of event ids for %s - hardware has only %d flag registers per event",
                        ToString(evt), EVENT_ID_NUM);
        return INVALID_TEVENTID;
    }

    template <HardEvent evt>
    __aicore__ inline void ReleaseEventID(TEventID id)
    {
        if (id < 0 || id >= EVENT_ID_NUM) { return; }
        eventOccupy_[static_cast<uint8_t>(evt)] &= static_cast<uint16_t>(~(1u << id));
    }

    /// Reset the allocators without touching data; lets a kernel reuse the
    /// whole of UB for a second phase.
    __aicore__ inline void Reset()
    {
        for (uint8_t h = 0; h < static_cast<uint8_t>(Hardware::MAX); ++h) { bufPool_[h].maxAddr = 0; }
        curBufSize_ = 0;
    }

    /// Drain: every buffer must be back to FREE and every flag consumed, or the
    /// kernel left the core in a state the next kernel would trip over.
    __aicore__ inline void Destroy()
    {
        if (destroyed_) { return; }
        destroyed_ = true;
        for (uint8_t i = 0; i < curBufSize_; ++i) {
            if (buf_[i].state != TBufState::FREE) {
                ReportSyncError("TPipe destroyed while buffer %s[0x%05x] is still %s - a tensor "
                                "was never freed", ToString(buf_[i].hard), buf_[i].address,
                                ToString(buf_[i].state));
            }
            if (buf_[i].freeBufEvtID != INVALID_TEVENTID) {
                // Mirrors FreeAllEvent(): consume the outstanding recycle flags.
                buf_[i].freeBufEvtID = INVALID_TEVENTID;
            }
        }
        // The recycle flags raised by the last FreeTensor of each buffer are
        // never waited on by anyone - real TPipe drains them in FreeAllEvent,
        // and so do we. A buffer left in ENQUE is the real leak, and the loop
        // above already reported it.
        CoreState &c = Core();
        for (uint8_t e = 0; e < EVENT_NUM; ++e) {
            for (int32_t id = 0; id < EVENT_ID_NUM; ++id) { c.flags[e][id].pending = false; }
        }
        if (TPipePtrRef() == this) { TPipePtrRef() = nullptr; }
    }

    __aicore__ inline uint32_t GetUsed(Hardware hard) const
    {
        return bufPool_[static_cast<uint8_t>(hard)].maxAddr;
    }

private:
    struct TPipeBufPool {
        uint32_t maxAddr = 0;
    };

    TPipeBufPool bufPool_[static_cast<uint8_t>(Hardware::MAX)];
    uint16_t eventOccupy_[EVENT_NUM] = {};
    TBufType buf_[QBUF_MAX_LEN];
    uint8_t curBufSize_ = 0;
    bool destroyed_ = false;
};

inline TPipe *&TPipePtrRef()
{
    static thread_local TPipe *pipe = nullptr;
    return pipe;
}

inline TPipe *GetTPipePtr() { return TPipePtrRef(); }

//===----------------------------------------------------------------------===//
// TQueBind - the queue, parameterised by the edge it spans
//===----------------------------------------------------------------------===//

template <TPosition src, TPosition dst, int32_t depth>
class TQueBind {
public:
    static constexpr TPosition srcPosition = src;
    static constexpr TPosition dstPosition = dst;
    static constexpr Hardware srcHardType = GetPhyType(src);
    static constexpr Hardware dstHardType = GetPhyType(dst);
    static constexpr Hardware bufHardType = GetBufferPos(src, dst);
    static constexpr TPosition position = GetPosition(src, dst);
    static constexpr int32_t queDepth = depth;

    // The two events that make the hand-over safe, picked by the same table the
    // real compiler uses. For VECIN (GM->UB) they are MTE2_V and V_MTE2; for
    // VECOUT (UB->GM) they are V_MTE3 and MTE3_V.
    static constexpr HardEvent enQueEvt = GetQueEvt(srcHardType, dstHardType, true);
    static constexpr HardEvent freeBufEvt = GetQueEvt(srcHardType, dstHardType, false);

    static_assert(depth >= 1, "TQue depth must be at least 1");
    static_assert(enQueEvt != HardEvent::MAX, "no hardware event for this src->dst pair");

    __aicore__ inline TQueBind() {}

    __aicore__ inline void InitBufHandles(TBufType *bufStart, uint8_t num)
    {
        bufStart_ = bufStart;
        bufNum_ = num;
        bufCursor_ = 0;
        head_ = 0;
        tail_ = 0;
        usedCount_ = 0;
    }

    /// Take a free slot. If the slot was released by the consumer we must first
    /// wait for that release to actually have happened on the other pipe.
    template <typename T>
    __aicore__ inline LocalTensor<T> AllocTensor()
    {
        TBufHandle buf = AllocBuffer();
        return Buf2Tensor<T>(buf);
    }

    /// Publish a filled buffer. The flag is raised on the producer pipe and
    /// carries everything that pipe has done so far.
    template <typename T>
    __aicore__ inline bool EnQue(const LocalTensor<T> &tensor)
    {
        return EnQue(tensor.GetBufferHandle());
    }

    __aicore__ inline bool EnQue(TBufHandle buf)
    {
        if (buf == nullptr) { return false; }
        if (buf->state != TBufState::OCCUPIED) {
            ReportSyncError("EnQue on a buffer in state %s (expected OCCUPIED - did you EnQue a "
                            "tensor you never allocated?)", ToString(buf->state));
            return false;
        }
        if (usedCount_ >= depth) {
            ReportSyncError("EnQue overflows the queue: depth is %d", depth);
            return false;
        }
        buf->state = TBufState::ENQUE;
        buf->enQueEvtID = GetTPipePtr()->template AllocEventID<enQueEvt>();
        SetFlag<enQueEvt>(buf->enQueEvtID);
        queue_[tail_] = buf;
        tail_ = static_cast<uint8_t>((tail_ + 1) % depth);
        usedCount_++;
        return true;
    }

    /// Take the oldest published buffer, blocking the consumer pipe until the
    /// producer's flag arrives.
    template <typename T>
    __aicore__ inline LocalTensor<T> DeQue()
    {
        return Buf2Tensor<T>(DeQue());
    }

    __aicore__ inline TBufHandle DeQue()
    {
        if (usedCount_ == 0) {
            ReportSyncError("DeQue on an empty queue - on hardware this reads a stale buffer");
            return nullptr;
        }
        TBufHandle buf = queue_[head_];
        head_ = static_cast<uint8_t>((head_ + 1) % depth);
        usedCount_--;
        if (buf->state != TBufState::ENQUE) {
            ReportSyncError("DeQue on a buffer in state %s (expected ENQUE)", ToString(buf->state));
            return buf;
        }
        buf->state = TBufState::DEQUE;
        WaitFlag<enQueEvt>(buf->enQueEvtID);
        GetTPipePtr()->template ReleaseEventID<enQueEvt>(buf->enQueEvtID);
        buf->enQueEvtID = INVALID_TEVENTID;
        return buf;
    }

    /// Give the slot back. The flag raised here is what a later AllocTensor
    /// waits on - the back edge that makes the ring safe to wrap around.
    template <typename T>
    __aicore__ inline void FreeTensor(LocalTensor<T> &tensor)
    {
        FreeBuffer(tensor.GetBufferHandle());
    }

    __aicore__ inline void FreeBuffer(TBufHandle buf)
    {
        if (buf == nullptr) { return; }
        if (buf->state == TBufState::FREE) {
            ReportSyncError("FreeTensor on a buffer that is already FREE (double free)");
            return;
        }
        buf->freeBufEvtID = GetTPipePtr()->template AllocEventID<freeBufEvt>();
        SetFlag<freeBufEvt>(buf->freeBufEvtID);
        buf->state = TBufState::FREE;
    }

    __aicore__ inline bool HasTensorInQue() const { return usedCount_ > 0; }
    __aicore__ inline int32_t GetTensorCountInQue() const { return usedCount_; }
    __aicore__ inline bool VacantInQue() const { return usedCount_ < depth; }
    __aicore__ inline bool HasIdleBuffer() const
    {
        for (uint8_t i = 0; i < bufNum_; ++i) {
            if (bufStart_[i].state == TBufState::FREE) { return true; }
        }
        return false;
    }

    static __aicore__ inline const char *PositionName()
    {
        switch (position) {
            case TPosition::VECIN:   return "VECIN";
            case TPosition::VECOUT:  return "VECOUT";
            case TPosition::VECCALC: return "VECCALC";
            case TPosition::A1:      return "A1";
            case TPosition::A2:      return "A2";
            case TPosition::B1:      return "B1";
            case TPosition::B2:      return "B2";
            case TPosition::CO1:     return "CO1";
            case TPosition::CO2:     return "CO2";
            default:                 return "?";
        }
    }

protected:
    __aicore__ inline TBufHandle AllocBuffer()
    {
        for (uint8_t i = 0; i < bufNum_; ++i) {
            TBufType *ret = bufStart_ + bufCursor_;
            bufCursor_ = static_cast<uint8_t>((bufCursor_ + 1) % bufNum_);
            if (ret->state != TBufState::FREE) { continue; }
            ret->state = TBufState::OCCUPIED;
            if (ret->freeBufEvtID != INVALID_TEVENTID) {
                WaitFlag<freeBufEvt>(ret->freeBufEvtID);
                GetTPipePtr()->template ReleaseEventID<freeBufEvt>(ret->freeBufEvtID);
                ret->freeBufEvtID = INVALID_TEVENTID;
            }
            return ret;
        }
        ReportSyncError("AllocTensor: all %u buffers of this %s queue are outstanding - free one "
                        "or give the queue more buffers", (unsigned)bufNum_, PositionName());
        return nullptr;
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> Buf2Tensor(TBufHandle buf)
    {
        TBuffAddr addr;
        if (buf != nullptr) {
            addr.logicPos = buf->logicPos;
            addr.bufferHandle = buf;
            addr.bufferAddr = buf->address;
            addr.dataLen = buf->dataLen;
        }
        return LocalTensor<T>(addr);
    }

    TBufType *bufStart_ = nullptr;
    uint8_t bufNum_ = 0;
    uint8_t bufCursor_ = 0;
    TBufHandle queue_[depth] = {};
    uint8_t head_ = 0;
    uint8_t tail_ = 0;
    int32_t usedCount_ = 0;
};

//===----------------------------------------------------------------------===//
// TQue and TBuf
//===----------------------------------------------------------------------===//

/// The queue as users declare it: one position, one depth. The position expands
/// into the src->dst edge, which picks the events.
template <TPosition pos, int32_t depth>
class TQue : public TQueBind<GetBufferLogicPos(pos, true), GetBufferLogicPos(pos, false), depth> {
public:
    __aicore__ inline TQue() = default;
};

/// Scratch memory with no queue and no events: you get the same buffer every
/// time, and any cross-pipe ordering is yours to write. Useful for a temporary
/// that only one pipe ever touches - and instructive when it isn't.
template <TPosition pos = TPosition::VECCALC>
class TBuf {
public:
    static constexpr Hardware bufHardType = GetPhyType(pos);
    static constexpr TPosition position = pos;
    static constexpr int32_t queDepth = 1;

    __aicore__ inline TBuf() = default;

    __aicore__ inline void InitBufHandles(TBufType *bufStart, uint8_t num)
    {
        bufStart_ = bufStart;
        (void)num;
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> Get() const
    {
        return Get<T>(bufStart_ == nullptr ? 0 : bufStart_->dataLen / sizeof(T));
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> Get(uint32_t len) const
    {
        TBuffAddr addr;
        if (bufStart_ != nullptr) {
            addr.logicPos = bufStart_->logicPos;
            addr.bufferHandle = bufStart_;
            addr.bufferAddr = bufStart_->address;
            addr.dataLen = len * sizeof(T);
        }
        return LocalTensor<T>(addr);
    }

    static __aicore__ inline const char *PositionName() { return "VECCALC"; }

private:
    TBufType *bufStart_ = nullptr;
};

template <TPosition pos>
__aicore__ inline bool TPipe::InitBuffer(TBuf<pos> &buf, uint32_t len)
{
    return InitBuffer<TBuf<pos>>(buf, 1, len);
}

}  // namespace AscendC

#endif  // MINI_ASCENDC_KERNEL_TPIPE_H
