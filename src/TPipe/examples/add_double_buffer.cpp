//===- add_double_buffer.cpp - the same add, tiled and pipelined ----------===//
//
// A core's share of the data rarely fits in UB, so the kernel walks it in
// tiles and the three stages become a software pipeline:
//
//   CopyIn(i)   MTE2 : GM -> UB
//   Compute(i)  V    : UB -> UB
//   CopyOut(i)  MTE3 : UB -> GM
//
// With BUFFER_NUM == 1 there is one buffer per queue, so CopyIn(i+1) cannot
// start until Compute(i) has released it: in the trace, every DataCopy is
// preceded by a WaitFlag<V_MTE2> for the tile that just finished, and the pipes
// take turns. With BUFFER_NUM == 2 the queue hands out the *other* buffer and
// that wait refers to tile i-1, so MTE2 runs a whole tile ahead of V.
//
// The kernel itself is in add_kernel.inc, included once per BUFFER_NUM.
//
//===----------------------------------------------------------------------===//
#include "mini_ascendc/kernel_operator.h"
#include "data_utils.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

namespace single_buffer {
constexpr int32_t BUFFER_NUM = 1;
#include "add_kernel.inc"
}  // namespace single_buffer

namespace double_buffer {
constexpr int32_t BUFFER_NUM = 2;
#include "add_kernel.inc"
}  // namespace double_buffer

namespace {

/// One core, a few small tiles, full trace - small enough to read.
template <typename Kernel>
void TraceOneCore(const char *title, Kernel kernel, uint32_t totalLength, uint32_t tileNum)
{
    std::vector<float> x = MakeInput(totalLength, 1);
    std::vector<float> y = MakeInput(totalLength, 2);
    std::vector<float> z(totalLength, 0.0f);

    std::printf("=== %s, %u tiles of %u floats on one core ===\n", title, tileNum,
                totalLength / tileNum);
    AscendC::SetTrace(true, 0);
    AscendC::KernelLaunch(1, kernel, reinterpret_cast<uint8_t *>(x.data()),
                          reinterpret_cast<uint8_t *>(y.data()),
                          reinterpret_cast<uint8_t *>(z.data()), totalLength, tileNum);
    AscendC::SetTrace(false);
    std::printf("  %llu SetFlag / %llu WaitFlag, %llu B in, %llu B out\n",
                (unsigned long long)AscendC::Core().setFlagCount,
                (unsigned long long)AscendC::Core().waitFlagCount,
                (unsigned long long)AscendC::Core().bytesIn,
                (unsigned long long)AscendC::Core().bytesOut);
    Verify("tiles", z, AddReference(x, y));
    std::printf("\n");
}

}  // namespace

int32_t main()
{
    // Small enough that the whole pipeline fits on screen. Watch where the
    // WaitFlag<V_MTE2> before each DataCopy lands relative to the Add.
    TraceOneCore("BUFFER_NUM = 1", single_buffer::add_custom, 3 * 64, 3);
    TraceOneCore("BUFFER_NUM = 2", double_buffer::add_custom, 3 * 64, 3);

    // The real run: 8 cores, 16 tiles each, double buffered.
    constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
    constexpr uint32_t BLOCK_NUM = 8;
    constexpr uint32_t TILE_NUM = 16;
    std::vector<float> x = MakeInput(TOTAL_LENGTH, 7);
    std::vector<float> y = MakeInput(TOTAL_LENGTH, 8);
    std::vector<float> z(TOTAL_LENGTH, 0.0f);

    std::printf("=== %u cores x %u tiles of %u floats, BUFFER_NUM = 2 ===\n", BLOCK_NUM, TILE_NUM,
                TOTAL_LENGTH / BLOCK_NUM / TILE_NUM);
    AscendC::KernelLaunch(BLOCK_NUM, double_buffer::add_custom,
                          reinterpret_cast<uint8_t *>(x.data()),
                          reinterpret_cast<uint8_t *>(y.data()),
                          reinterpret_cast<uint8_t *>(z.data()), TOTAL_LENGTH, TILE_NUM);

    bool ok = Verify("add_double_buffer", z, AddReference(x, y));
    uint32_t errors = AscendC::GetSyncErrorCount();
    std::printf("  sync errors: %u\n", errors);
    return (ok && errors == 0) ? 0 : 1;
}
