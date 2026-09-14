//===- kernel_common.h - positions, pipes, events, the simulated core -----===//
//
// The machine model everything else is built on:
//
//   * TPosition / Hardware   - where a tensor lives (logical vs. physical)
//   * PipeId                 - the asynchronous instruction queues of one core
//   * HardEvent              - a (src pipe -> dst pipe) synchronisation edge
//   * CoreState              - one AI core: its local memories, its pipe clocks
//
// On real hardware the pipes run concurrently and SetFlag/WaitFlag are the only
// thing that orders them. Here everything executes in program order, so instead
// of *simulating* the concurrency we *check* it: every pipe carries a vector
// clock, every memory write is tagged with (pipe, clock), and every access from
// a different pipe must be covered by a WaitFlag that observed that clock.
// A kernel that forgets a flag runs fine but is reported - which is exactly the
// bug class TQue exists to make impossible.
//
//===----------------------------------------------------------------------===//
#ifndef MINI_ASCENDC_KERNEL_COMMON_H
#define MINI_ASCENDC_KERNEL_COMMON_H

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Ascend C decorates kernels and pointers; on the host they are all no-ops, but
// keeping them means example kernels read like the real thing.
#define __aicore__
#define __global__
#define __gm__
#define __ubuf__

namespace AscendC {

//===----------------------------------------------------------------------===//
// Positions and physical memories
//===----------------------------------------------------------------------===//

/// Logical position of a tensor. This is the vocabulary the programmer uses;
/// the compiler maps it to a physical memory (Hardware) and to a pipe.
enum class TPosition : int32_t {
    GM,
    A1, A2, B1, B2, C1, C2, CO1, CO2,
    VECIN, VECOUT, VECCALC,
    LCM = VECCALC,
    SPM,
    TSCM,
    MAX,
};

/// Physical memory. UB (unified buffer) is the vector core's scratchpad; L1 and
/// L0A/L0B/L0C belong to the cube path.
enum class Hardware : uint8_t { GM, UB, L1, L0A, L0B, L0C, MAX };

/// The independently scheduled instruction queues inside one core.
///   S    scalar          MTE1  L1  -> L0A/L0B/UB
///   V    vector          MTE2  GM  -> UB/L1/L0
///   M    cube (mmad)     MTE3  UB  -> GM/L1
enum class PipeId : uint8_t { S, V, M, MTE1, MTE2, MTE3, MAX };

constexpr uint8_t PIPE_NUM = static_cast<uint8_t>(PipeId::MAX);

inline const char *ToString(PipeId p)
{
    switch (p) {
        case PipeId::S:    return "S";
        case PipeId::V:    return "V";
        case PipeId::M:    return "M";
        case PipeId::MTE1: return "MTE1";
        case PipeId::MTE2: return "MTE2";
        case PipeId::MTE3: return "MTE3";
        default:           return "?";
    }
}

inline const char *ToString(Hardware h)
{
    switch (h) {
        case Hardware::GM:  return "gm";
        case Hardware::UB:  return "ub";
        case Hardware::L1:  return "l1";
        case Hardware::L0A: return "l0a";
        case Hardware::L0B: return "l0b";
        case Hardware::L0C: return "l0c";
        default:            return "?";
    }
}

/// TPosition -> the memory it actually sits in.
constexpr Hardware GetPhyType(TPosition pos)
{
    switch (pos) {
        case TPosition::GM:   return Hardware::GM;
        case TPosition::A1:
        case TPosition::B1:
        case TPosition::C1:
        case TPosition::TSCM: return Hardware::L1;
        case TPosition::A2:   return Hardware::L0A;
        case TPosition::B2:   return Hardware::L0B;
        case TPosition::CO1:  return Hardware::L0C;
        default:              return Hardware::UB;  // CO2/VECIN/VECOUT/VECCALC/SPM
    }
}

/// A TQue is declared with one position but describes an *edge*: data enters it
/// from somewhere and leaves it for somewhere else. VECIN means "GM -> UB",
/// VECOUT means "UB -> GM". Same table as AscendC::GetBufferLogicPos.
constexpr TPosition GetBufferLogicPos(TPosition pos, bool isSrc)
{
    switch (pos) {
        case TPosition::A1:     return isSrc ? TPosition::GM     : TPosition::A1;
        case TPosition::B1:     return isSrc ? TPosition::GM     : TPosition::B1;
        case TPosition::C1:     return isSrc ? TPosition::GM     : TPosition::C1;
        case TPosition::A2:     return isSrc ? TPosition::A1     : TPosition::A2;
        case TPosition::B2:     return isSrc ? TPosition::B1     : TPosition::B2;
        case TPosition::C2:     return isSrc ? TPosition::C1     : TPosition::C2;
        case TPosition::CO1:    return isSrc ? TPosition::CO1    : TPosition::CO2;
        case TPosition::CO2:    return isSrc ? TPosition::CO2    : TPosition::GM;
        case TPosition::VECIN:  return isSrc ? TPosition::GM     : TPosition::VECIN;
        case TPosition::VECOUT: return isSrc ? TPosition::VECOUT : TPosition::GM;
        case TPosition::SPM:    return isSrc ? TPosition::VECOUT : TPosition::GM;
        default:                return TPosition::MAX;
    }
}

/// Where the buffer of a src->dst queue is carved from: the non-GM end.
constexpr TPosition GetPosition(TPosition src, TPosition dst)
{
    return (src == TPosition::GM) ? dst : src;
}

constexpr Hardware GetBufferPos(TPosition src, TPosition dst)
{
    return GetPhyType(GetPosition(src, dst));
}

//===----------------------------------------------------------------------===//
// Hardware events
//===----------------------------------------------------------------------===//

// Every event is named after the edge it guards: producer pipe -> consumer pipe.
#define MINI_EVENT_LIST(X)   \
    X(MTE2_V,    MTE2, V)    \
    X(V_MTE2,    V,    MTE2) \
    X(V_MTE3,    V,    MTE3) \
    X(MTE3_V,    MTE3, V)    \
    X(MTE2_MTE1, MTE2, MTE1) \
    X(MTE1_MTE2, MTE1, MTE2) \
    X(MTE1_M,    MTE1, M)    \
    X(M_MTE1,    M,    MTE1) \
    X(MTE1_V,    MTE1, V)    \
    X(V_MTE1,    V,    MTE1) \
    X(MTE2_M,    MTE2, M)    \
    X(M_MTE2,    M,    MTE2) \
    X(M_V,       M,    V)    \
    X(V_M,       V,    M)    \
    X(S_V,       S,    V)    \
    X(V_S,       V,    S)    \
    X(S_MTE2,    S,    MTE2) \
    X(MTE2_S,    MTE2, S)    \
    X(S_MTE3,    S,    MTE3) \
    X(MTE3_S,    MTE3, S)    \
    X(MTE2_MTE3, MTE2, MTE3) \
    X(MTE3_MTE2, MTE3, MTE2) \
    X(MTE3_MTE1, MTE3, MTE1) \
    X(MTE1_MTE3, MTE1, MTE3)

enum class HardEvent : uint8_t {
#define X(name, s, d) name,
    MINI_EVENT_LIST(X)
#undef X
    MAX,
};

constexpr uint8_t EVENT_NUM = static_cast<uint8_t>(HardEvent::MAX);
/// Hardware has a small, fixed number of flag registers per event type.
constexpr int32_t EVENT_ID_NUM = 8;

using TEventID = int8_t;
constexpr TEventID INVALID_TEVENTID = -1;

constexpr PipeId EventSrcPipe(HardEvent e)
{
    switch (e) {
#define X(name, s, d) case HardEvent::name: return PipeId::s;
        MINI_EVENT_LIST(X)
#undef X
        default: return PipeId::MAX;
    }
}

constexpr PipeId EventDstPipe(HardEvent e)
{
    switch (e) {
#define X(name, s, d) case HardEvent::name: return PipeId::d;
        MINI_EVENT_LIST(X)
#undef X
        default: return PipeId::MAX;
    }
}

inline const char *ToString(HardEvent e)
{
    switch (e) {
#define X(name, s, d) case HardEvent::name: return #name;
        MINI_EVENT_LIST(X)
#undef X
        default: return "MAX";
    }
}

/// The event a queue uses. fwdDirect == true is the producer handing the buffer
/// over (EnQue/DeQue); false is the consumer giving it back (FreeTensor/Alloc).
/// Mirrors AscendC::GetQueEvt.
constexpr HardEvent GetQueEvt(Hardware src, Hardware dst, bool fwdDirect)
{
    if (src == Hardware::GM) {                     // filled by MTE2
        if (dst == Hardware::UB)  { return fwdDirect ? HardEvent::MTE2_V : HardEvent::V_MTE2; }
        if (dst == Hardware::L1)  { return fwdDirect ? HardEvent::MTE2_MTE1 : HardEvent::MTE1_MTE2; }
        if (dst == Hardware::L0A || dst == Hardware::L0B) {
            return fwdDirect ? HardEvent::MTE2_M : HardEvent::M_MTE2;
        }
    } else if (src == Hardware::UB) {              // drained by MTE3
        if (dst == Hardware::GM)  { return fwdDirect ? HardEvent::V_MTE3 : HardEvent::MTE3_V; }
        if (dst == Hardware::L1)  { return fwdDirect ? HardEvent::MTE3_MTE1 : HardEvent::MTE1_MTE3; }
        if (dst == Hardware::UB)  { return fwdDirect ? HardEvent::MTE2_MTE3 : HardEvent::MTE3_MTE2; }
    } else if (src == Hardware::L1) {              // moved on by MTE1
        if (dst == Hardware::L0A || dst == Hardware::L0B) {
            return fwdDirect ? HardEvent::MTE1_M : HardEvent::M_MTE1;
        }
        if (dst == Hardware::UB)  { return fwdDirect ? HardEvent::MTE1_V : HardEvent::V_MTE1; }
    } else if (src == Hardware::L0C) {             // cube result read by vector
        if (dst == Hardware::UB)  { return fwdDirect ? HardEvent::M_V : HardEvent::V_M; }
    }
    return HardEvent::MAX;
}

//===----------------------------------------------------------------------===//
// One simulated AI core
//===----------------------------------------------------------------------===//

/// Real sizes of an Ascend vector/cube core's local memories.
constexpr uint32_t TOTAL_UB_SIZE  = 192 * 1024;
constexpr uint32_t TOTAL_L1_SIZE  = 512 * 1024;
constexpr uint32_t TOTAL_L0A_SIZE = 64 * 1024;
constexpr uint32_t TOTAL_L0B_SIZE = 64 * 1024;
constexpr uint32_t TOTAL_L0C_SIZE = 128 * 1024;

/// All DMA on Ascend moves whole 32-byte blocks.
constexpr uint32_t ONE_BLK_SIZE = 32;

constexpr uint32_t HardwareSize(Hardware h)
{
    switch (h) {
        case Hardware::UB:  return TOTAL_UB_SIZE;
        case Hardware::L1:  return TOTAL_L1_SIZE;
        case Hardware::L0A: return TOTAL_L0A_SIZE;
        case Hardware::L0B: return TOTAL_L0B_SIZE;
        case Hardware::L0C: return TOTAL_L0C_SIZE;
        default:            return 0;
    }
}

enum class SyncCheck : uint8_t { Off, Report, Abort };

/// Who last touched a 32-byte granule, and at which tick of that pipe's clock.
/// Last writer catches read-after-write and write-after-write; last reader
/// catches write-after-read (the bug double buffering exists to create).
struct Granule {
    uint8_t writer;
    uint64_t clock;
    uint8_t reader;
    uint64_t readClock;
};

struct EventSlot {
    bool pending;
    uint64_t snapshot[PIPE_NUM];
};

struct CoreState {
    uint32_t blockIdx = 0;
    uint32_t blockNum = 1;

    std::vector<uint8_t> mem[static_cast<uint8_t>(Hardware::MAX)];
    std::vector<Granule> gran[static_cast<uint8_t>(Hardware::MAX)];

    // Vector clocks. clock[p] ticks once per instruction issued on pipe p;
    // view[p][q] is how much of pipe q's history pipe p is known to observe.
    uint64_t clock[PIPE_NUM] = {};
    uint64_t view[PIPE_NUM][PIPE_NUM] = {};
    EventSlot flags[EVENT_NUM][EVENT_ID_NUM] = {};

    SyncCheck checkMode = SyncCheck::Report;
    bool trace = false;
    int32_t traceBlock = 0;       // -1 = every block
    uint32_t syncErrors = 0;
    uint64_t setFlagCount = 0;
    uint64_t waitFlagCount = 0;
    uint64_t bytesIn = 0;
    uint64_t bytesOut = 0;

    void Reset(uint32_t idx, uint32_t num)
    {
        blockIdx = idx;
        blockNum = num;
        for (uint8_t h = 0; h < static_cast<uint8_t>(Hardware::MAX); ++h) {
            uint32_t size = HardwareSize(static_cast<Hardware>(h));
            if (size == 0) { continue; }
            mem[h].assign(size, 0);
            gran[h].assign(size / ONE_BLK_SIZE,
                           Granule{static_cast<uint8_t>(PipeId::MAX), 0,
                                   static_cast<uint8_t>(PipeId::MAX), 0});
        }
        std::memset(clock, 0, sizeof(clock));
        std::memset(view, 0, sizeof(view));
        std::memset(flags, 0, sizeof(flags));
        setFlagCount = waitFlagCount = bytesIn = bytesOut = 0;
    }

    uint8_t *Base(Hardware h) { return mem[static_cast<uint8_t>(h)].data(); }
};

inline CoreState &Core()
{
    static thread_local CoreState state;
    return state;
}

inline uint32_t GetBlockIdx() { return Core().blockIdx; }
inline uint32_t GetBlockNum() { return Core().blockNum; }

//===----------------------------------------------------------------------===//
// Trace
//===----------------------------------------------------------------------===//

/// Print every pipe instruction as it issues. `block` selects which core to
/// trace (-1 for all); tracing all 8 cores of a launch is rarely what you want.
inline void SetTrace(bool on, int32_t block = 0)
{
    Core().trace = on;
    Core().traceBlock = block;
}

inline bool Tracing()
{
    CoreState &c = Core();
    return c.trace && (c.traceBlock < 0 || static_cast<uint32_t>(c.traceBlock) == c.blockIdx);
}

inline void Trace(PipeId pipe, const char *fmt, ...)
{
    if (!Tracing()) { return; }
    std::printf("  [core %u][%-4s] ", Core().blockIdx, ToString(pipe));
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
}

//===----------------------------------------------------------------------===//
// The race checker
//===----------------------------------------------------------------------===//

inline void ReportSyncError(const char *fmt, ...)
{
    CoreState &c = Core();
    if (c.checkMode == SyncCheck::Off) { return; }
    c.syncErrors++;
    std::printf("  !! sync error (core %u): ", c.blockIdx);
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
    if (c.checkMode == SyncCheck::Abort) { std::abort(); }
}

inline void SetSyncCheck(SyncCheck mode) { Core().checkMode = mode; }
inline uint32_t GetSyncErrorCount() { return Core().syncErrors; }
inline void ClearSyncErrors() { Core().syncErrors = 0; }

/// The event name a programmer would have to use to fix a missing edge.
inline const char *SuggestEvent(PipeId writer, PipeId reader)
{
    for (uint8_t e = 0; e < EVENT_NUM; ++e) {
        HardEvent evt = static_cast<HardEvent>(e);
        if (EventSrcPipe(evt) == writer && EventDstPipe(evt) == reader) { return ToString(evt); }
    }
    return "<no such event>";
}

/// One tick of `pipe`'s clock; every instruction calls this.
inline uint64_t PipeTick(PipeId pipe)
{
    return ++Core().clock[static_cast<uint8_t>(pipe)];
}

inline void CheckAccess(Hardware hw, uint32_t addr, uint32_t bytes, PipeId pipe, const char *what,
                        bool isWrite = false)
{
    CoreState &c = Core();
    if (c.checkMode == SyncCheck::Off || hw == Hardware::GM) { return; }
    auto &g = c.gran[static_cast<uint8_t>(hw)];
    uint32_t first = addr / ONE_BLK_SIZE;
    uint32_t last = (addr + bytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE;
    uint8_t me = static_cast<uint8_t>(pipe);
    for (uint32_t i = first; i < last && i < g.size(); ++i) {
        uint8_t w = g[i].writer;
        if (w != static_cast<uint8_t>(PipeId::MAX) && w != me && c.view[me][w] < g[i].clock) {
            ReportSyncError("%s on pipe %s touches %s[0x%05x] last written by %s at tick %llu, "
                            "but %s has only observed %s up to tick %llu - missing WaitFlag<%s>",
                            what, ToString(pipe), ToString(hw), i * ONE_BLK_SIZE,
                            ToString(static_cast<PipeId>(w)), (unsigned long long)g[i].clock,
                            ToString(pipe), ToString(static_cast<PipeId>(w)),
                            (unsigned long long)c.view[me][w],
                            SuggestEvent(static_cast<PipeId>(w), pipe));
            return;  // one report per access is enough
        }
        uint8_t r = g[i].reader;
        if (isWrite && r != static_cast<uint8_t>(PipeId::MAX) && r != me &&
            c.view[me][r] < g[i].readClock) {
            ReportSyncError("%s on pipe %s overwrites %s[0x%05x] still being read by %s "
                            "(read at tick %llu, %s has observed %s only up to tick %llu) - "
                            "missing WaitFlag<%s>",
                            what, ToString(pipe), ToString(hw), i * ONE_BLK_SIZE,
                            ToString(static_cast<PipeId>(r)), (unsigned long long)g[i].readClock,
                            ToString(pipe), ToString(static_cast<PipeId>(r)),
                            (unsigned long long)c.view[me][r],
                            SuggestEvent(static_cast<PipeId>(r), pipe));
            return;
        }
    }
}

inline void RecordRead(Hardware hw, uint32_t addr, uint32_t bytes, PipeId pipe, uint64_t tick)
{
    CoreState &c = Core();
    if (c.checkMode == SyncCheck::Off || hw == Hardware::GM) { return; }
    auto &g = c.gran[static_cast<uint8_t>(hw)];
    uint32_t first = addr / ONE_BLK_SIZE;
    uint32_t last = (addr + bytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE;
    for (uint32_t i = first; i < last && i < g.size(); ++i) {
        g[i].reader = static_cast<uint8_t>(pipe);
        g[i].readClock = tick;
    }
}

inline void RecordWrite(Hardware hw, uint32_t addr, uint32_t bytes, PipeId pipe, uint64_t tick)
{
    CoreState &c = Core();
    if (hw == Hardware::GM) { return; }
    auto &g = c.gran[static_cast<uint8_t>(hw)];
    if (c.checkMode == SyncCheck::Off) { return; }
    uint32_t first = addr / ONE_BLK_SIZE;
    uint32_t last = (addr + bytes + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE;
    for (uint32_t i = first; i < last && i < g.size(); ++i) {
        g[i].writer = static_cast<uint8_t>(pipe);
        g[i].clock = tick;
        g[i].reader = static_cast<uint8_t>(PipeId::MAX);
        g[i].readClock = 0;
    }
}

//===----------------------------------------------------------------------===//
// SetFlag / WaitFlag / PipeBarrier
//===----------------------------------------------------------------------===//

/// Raise flag `id` of this event: the producer pipe publishes everything it has
/// issued so far. Non-blocking on hardware, and non-blocking here.
template <HardEvent evt>
inline void SetFlag(TEventID id)
{
    constexpr PipeId src = EventSrcPipe(evt);
    CoreState &c = Core();
    if (id < 0 || id >= EVENT_ID_NUM) {
        ReportSyncError("SetFlag<%s> with invalid event id %d", ToString(evt), (int)id);
        return;
    }
    EventSlot &slot = c.flags[static_cast<uint8_t>(evt)][id];
    if (slot.pending) {
        ReportSyncError("SetFlag<%s>(%d) raised twice without an intervening WaitFlag",
                        ToString(evt), (int)id);
    }
    uint8_t s = static_cast<uint8_t>(src);
    std::memcpy(slot.snapshot, c.view[s], sizeof(slot.snapshot));
    slot.snapshot[s] = c.clock[s];
    slot.pending = true;
    c.setFlagCount++;
    Trace(src, "SetFlag<%s>(%d)", ToString(evt), (int)id);
}

/// Block the consumer pipe until flag `id` is raised, then inherit everything
/// the producer had published.
template <HardEvent evt>
inline void WaitFlag(TEventID id)
{
    constexpr PipeId dst = EventDstPipe(evt);
    CoreState &c = Core();
    if (id < 0 || id >= EVENT_ID_NUM) {
        ReportSyncError("WaitFlag<%s> with invalid event id %d", ToString(evt), (int)id);
        return;
    }
    EventSlot &slot = c.flags[static_cast<uint8_t>(evt)][id];
    if (!slot.pending) {
        ReportSyncError("WaitFlag<%s>(%d) with no matching SetFlag - on hardware this hangs",
                        ToString(evt), (int)id);
        return;
    }
    uint8_t d = static_cast<uint8_t>(dst);
    for (uint8_t p = 0; p < PIPE_NUM; ++p) {
        if (slot.snapshot[p] > c.view[d][p]) { c.view[d][p] = slot.snapshot[p]; }
    }
    slot.pending = false;
    c.waitFlagCount++;
    Trace(dst, "WaitFlag<%s>(%d)", ToString(evt), (int)id);
}

/// Order two instructions on the *same* pipe. Program order already gives us
/// that here, so it only shows up in the trace.
template <PipeId pipe>
inline void PipeBarrier()
{
    Trace(pipe, "PipeBarrier<%s>", ToString(pipe));
}

//===----------------------------------------------------------------------===//
// Kernel launch
//===----------------------------------------------------------------------===//

/// Stands in for `kernel<<<blockNum, ..., stream>>>(...)`. Blocks run one after
/// another, each on a freshly zeroed core - so a kernel that relies on UB
/// content left by another block breaks here, as it would on hardware.
template <typename Fn, typename... Args>
inline void KernelLaunch(uint32_t blockNum, Fn &&kernel, Args &&...args)
{
    for (uint32_t i = 0; i < blockNum; ++i) {
        CoreState &c = Core();
        bool trace = c.trace;
        int32_t traceBlock = c.traceBlock;
        SyncCheck mode = c.checkMode;
        uint32_t errors = c.syncErrors;
        c.Reset(i, blockNum);
        c.trace = trace;
        c.traceBlock = traceBlock;
        c.checkMode = mode;
        c.syncErrors = errors;
        kernel(args...);
    }
}

}  // namespace AscendC

#endif  // MINI_ASCENDC_KERNEL_COMMON_H
