//===- add_kernel.inc - the tiled add kernel, verbatim --------------------===//
//
// Included twice by add_double_buffer.cpp, once with BUFFER_NUM = 1 and once
// with BUFFER_NUM = 2. Not a template, not #ifdef'd: the two pipelines really
// are the same source text, and the only difference is that one constant.
//
//===----------------------------------------------------------------------===//

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        blockLength_ = totalLength / AscendC::GetBlockNum();
        tileNum_ = tileNum;
        tileLength_ = blockLength_ / tileNum;

        uint32_t offset = blockLength_ * AscendC::GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ float *)x + offset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ float *)y + offset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ float *)z + offset, blockLength_);

        // BUFFER_NUM buffers per queue: this single number is double buffering.
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(float));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileNum_; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /// MTE2 stage. AllocTensor is where the back pressure lives: with one
    /// buffer it waits for Compute to release this very buffer.
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[progress * tileLength_], tileLength_);
        AscendC::DataCopy(yLocal, yGm_[progress * tileLength_], tileLength_);
        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    /// V stage. Free the inputs as soon as the Add has read them - that is what
    /// lets the next CopyIn start.
    __aicore__ inline void Compute(uint32_t progress)
    {
        (void)progress;
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ_.AllocTensor<float>();
        AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
        outQueueZ_.EnQue<float>(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    /// MTE3 stage.
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<float> zLocal = outQueueZ_.DeQue<float>();
        AscendC::DataCopy(zGm_[progress * tileLength_], zLocal, tileLength_);
        outQueueZ_.FreeTensor(zLocal);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> zGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileLength_ = 0;
};

__global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z,
                                      uint32_t totalLength, uint32_t tileNum)
{
    KernelAdd op;
    op.Init(x, y, z, totalLength, tileNum);
    op.Process();
}
