//===- sync_check.cpp - what TQue is actually buying you ------------------===//
//
// The same add written four ways. Only the third one is what you would ship;
// the others exist to show what goes wrong without the queue, and what the
// checker says about it.
//
//   1. TBuf, no flags        - MTE2 fills a buffer, V reads it. On the CPU
//                              simulator the numbers come out right anyway,
//                              which is exactly why this bug survives testing.
//   2. TBuf, manual flags    - correct, and this is the code TQue writes for
//                              you: alloc an event id, SetFlag, WaitFlag.
//   3. TQue                  - the same thing, with the events implied.
//   4. TQue, missing Free    - the state machine catches a leaked buffer.
//
//===----------------------------------------------------------------------===//
#include "mini_ascendc/kernel_operator.h"
#include "data_utils.h"

using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

constexpr uint32_t LENGTH = 256;

namespace {

/// 1. Raw scratch buffers and no synchronisation at all.
__global__ __aicore__ void add_unsynced(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                        uint32_t len)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufX, bufY, bufZ;
    pipe.InitBuffer(bufX, len * sizeof(float));
    pipe.InitBuffer(bufY, len * sizeof(float));
    pipe.InitBuffer(bufZ, len * sizeof(float));

    GlobalTensor<float> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float *)x, len);
    yGm.SetGlobalBuffer((__gm__ float *)y, len);
    zGm.SetGlobalBuffer((__gm__ float *)z, len);

    LocalTensor<float> xLocal = bufX.Get<float>();
    LocalTensor<float> yLocal = bufY.Get<float>();
    LocalTensor<float> zLocal = bufZ.Get<float>();

    AscendC::DataCopy(xLocal, xGm, len);   // MTE2 writes xLocal
    AscendC::DataCopy(yLocal, yGm, len);   // MTE2 writes yLocal
    AscendC::Add(zLocal, xLocal, yLocal, len);  // V reads them - with no flag
    AscendC::DataCopy(zGm, zLocal, len);        // MTE3 reads zLocal - same problem
}

/// 2. The same thing done by hand. This is what TQue generates.
__global__ __aicore__ void add_manual_flags(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                            uint32_t len)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufX, bufY, bufZ;
    pipe.InitBuffer(bufX, len * sizeof(float));
    pipe.InitBuffer(bufY, len * sizeof(float));
    pipe.InitBuffer(bufZ, len * sizeof(float));

    GlobalTensor<float> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float *)x, len);
    yGm.SetGlobalBuffer((__gm__ float *)y, len);
    zGm.SetGlobalBuffer((__gm__ float *)z, len);

    LocalTensor<float> xLocal = bufX.Get<float>();
    LocalTensor<float> yLocal = bufY.Get<float>();
    LocalTensor<float> zLocal = bufZ.Get<float>();

    AscendC::DataCopy(xLocal, xGm, len);
    AscendC::DataCopy(yLocal, yGm, len);

    TEventID copyDone = pipe.AllocEventID<HardEvent::MTE2_V>();
    AscendC::SetFlag<HardEvent::MTE2_V>(copyDone);   // on MTE2
    AscendC::WaitFlag<HardEvent::MTE2_V>(copyDone);  // blocks V
    pipe.ReleaseEventID<HardEvent::MTE2_V>(copyDone);

    AscendC::Add(zLocal, xLocal, yLocal, len);

    TEventID addDone = pipe.AllocEventID<HardEvent::V_MTE3>();
    AscendC::SetFlag<HardEvent::V_MTE3>(addDone);    // on V
    AscendC::WaitFlag<HardEvent::V_MTE3>(addDone);   // blocks MTE3
    pipe.ReleaseEventID<HardEvent::V_MTE3>(addDone);

    AscendC::DataCopy(zGm, zLocal, len);
}

/// 3. With queues: no event appears in the source at all.
__global__ __aicore__ void add_queued(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                      uint32_t len)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, 1> outQueueZ;
    pipe.InitBuffer(inQueueX, 1, len * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, len * sizeof(float));
    pipe.InitBuffer(outQueueZ, 1, len * sizeof(float));

    GlobalTensor<float> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float *)x, len);
    yGm.SetGlobalBuffer((__gm__ float *)y, len);
    zGm.SetGlobalBuffer((__gm__ float *)z, len);

    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm, len);
    AscendC::DataCopy(yLocal, yGm, len);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);

    xLocal = inQueueX.DeQue<float>();
    yLocal = inQueueY.DeQue<float>();
    LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Add(zLocal, xLocal, yLocal, len);
    outQueueZ.EnQue(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);

    zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm, zLocal, len);
    outQueueZ.FreeTensor(zLocal);
}

/// 4. Correct synchronisation, but the output buffer is never given back.
__global__ __aicore__ void add_leaky(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                     uint32_t len)
{
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, 1> outQueueZ;
    pipe.InitBuffer(inQueueX, 1, len * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, len * sizeof(float));
    pipe.InitBuffer(outQueueZ, 1, len * sizeof(float));

    GlobalTensor<float> xGm, yGm, zGm;
    xGm.SetGlobalBuffer((__gm__ float *)x, len);
    yGm.SetGlobalBuffer((__gm__ float *)y, len);
    zGm.SetGlobalBuffer((__gm__ float *)z, len);

    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm, len);
    AscendC::DataCopy(yLocal, yGm, len);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);

    xLocal = inQueueX.DeQue<float>();
    yLocal = inQueueY.DeQue<float>();
    LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Add(zLocal, xLocal, yLocal, len);
    outQueueZ.EnQue(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);

    zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm, zLocal, len);
    // outQueueZ.FreeTensor(zLocal);  <- forgotten
}

template <typename Kernel>
uint32_t Run(const char *title, Kernel kernel, uint32_t expectedErrors)
{
    std::vector<float> x = MakeInput(LENGTH, 3);
    std::vector<float> y = MakeInput(LENGTH, 4);
    std::vector<float> z(LENGTH, 0.0f);

    std::printf("=== %s ===\n", title);
    AscendC::ClearSyncErrors();
    AscendC::KernelLaunch(1, kernel, reinterpret_cast<uint8_t *>(x.data()),
                          reinterpret_cast<uint8_t *>(y.data()),
                          reinterpret_cast<uint8_t *>(z.data()), LENGTH);
    uint32_t errors = AscendC::GetSyncErrorCount();
    bool valuesOk = Verify(title, z, AddReference(x, y));
    std::printf("  %u error(s) reported, expected %u -> %s\n\n", errors, expectedErrors,
                errors == expectedErrors ? "as designed" : "UNEXPECTED");
    return (valuesOk && errors == expectedErrors) ? 0 : 1;
}

}  // namespace

int32_t main()
{
    uint32_t bad = 0;
    // Two unsynchronised reads inside Add (x and y) plus one in the copy out.
    bad += Run("1. TBuf, no synchronisation", add_unsynced, 3);
    bad += Run("2. TBuf, manual SetFlag/WaitFlag", add_manual_flags, 0);
    bad += Run("3. TQue, synchronisation implied", add_queued, 0);
    // The leak is reported when TPipe is destroyed at the end of the kernel.
    bad += Run("4. TQue, FreeTensor forgotten", add_leaky, 1);

    std::printf("%s\n", bad == 0 ? "checker behaved as designed in all four cases"
                                 : "checker did NOT behave as designed");
    return bad == 0 ? 0 : 1;
}
