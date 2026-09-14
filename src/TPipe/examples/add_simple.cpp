//===- add_simple.cpp - vector add, one tile per core ---------------------===//
//
// The shortest kernel that uses the full TPipe/TQue programming model, and a
// line-for-line port of the official add_tpipe_tque sample. Each core takes
// 1/blockNum of the data, moves it GM -> UB, adds, and moves UB -> GM.
//
// Nothing in the kernel body mentions a flag or an event: the three queues
// insert MTE2_V / V_MTE2 / V_MTE3 / MTE3_V for us. Run it and read the trace.
//
//===----------------------------------------------------------------------===//
#include "mini_ascendc/kernel_operator.h"
#include "data_utils.h"

constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
constexpr uint32_t BLOCK_NUM = 8;

__global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                      uint32_t totalLength)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;

    // Split the problem by core: block_idx picks a disjoint slice of GM.
    uint32_t blockLength = totalLength / AscendC::GetBlockNum();
    uint32_t offset = blockLength * AscendC::GetBlockIdx();
    xGm.SetGlobalBuffer((__gm__ float *)x + offset, blockLength);
    yGm.SetGlobalBuffer((__gm__ float *)y + offset, blockLength);
    zGm.SetGlobalBuffer((__gm__ float *)z + offset, blockLength);

    // UB is carved up once, here. 3 x 8KB out of 192KB.
    pipe.InitBuffer(inQueueX, 1, blockLength * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, blockLength * sizeof(float));
    pipe.InitBuffer(outQueueZ, 1, blockLength * sizeof(float));

    // Stage 1 (MTE2): bring the inputs in and publish them.
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm, blockLength);
    AscendC::DataCopy(yLocal, yGm, blockLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);

    // Stage 2 (V): DeQue blocks the vector pipe until the copies landed.
    xLocal = inQueueX.DeQue<float>();
    yLocal = inQueueY.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Add(zLocal, xLocal, yLocal, blockLength);
    outQueueZ.EnQue<float>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);

    // Stage 3 (MTE3): write the result back.
    zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm, zLocal, blockLength);
    outQueueZ.FreeTensor(zLocal);
}

int32_t main()
{
    std::vector<float> x = MakeInput(TOTAL_LENGTH, 1);
    std::vector<float> y = MakeInput(TOTAL_LENGTH, 2);
    std::vector<float> z(TOTAL_LENGTH, 0.0f);

    std::printf("add_simple: %u floats over %u cores, %u floats per core\n\n", TOTAL_LENGTH,
                BLOCK_NUM, TOTAL_LENGTH / BLOCK_NUM);
    std::printf("trace of core 0 (every instruction, on the pipe that issues it):\n");
    AscendC::SetTrace(true, /*block=*/0);

    AscendC::KernelLaunch(BLOCK_NUM, add_custom, reinterpret_cast<uint8_t *>(x.data()),
                          reinterpret_cast<uint8_t *>(y.data()),
                          reinterpret_cast<uint8_t *>(z.data()), TOTAL_LENGTH);
    AscendC::SetTrace(false);

    std::printf("\n");
    bool ok = Verify("add_simple", z, AddReference(x, y));
    uint32_t errors = AscendC::GetSyncErrorCount();
    std::printf("  sync errors: %u\n", errors);
    return (ok && errors == 0) ? 0 : 1;
}
