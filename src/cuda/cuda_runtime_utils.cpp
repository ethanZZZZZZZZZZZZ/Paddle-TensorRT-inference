#include "cuda/cuda_runtime_utils.h"

#if defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)

#include <algorithm>
#include <sstream>
#include <utility>

#include "common/logging.h"

namespace edge {

CudaStreamLease::CudaStreamLease(cudaStream_t stream, int index)
    : stream_(stream),
      index_(index) {}

cudaStream_t CudaStreamLease::Get() const {
    return stream_;
}

int CudaStreamLease::Index() const {
    return index_;
}

bool CudaStreamLease::Valid() const {
    return stream_ != nullptr && index_ >= 0;
}

CudaStreamPool::~CudaStreamPool() {
    Reset();
}

bool CudaStreamPool::Init(int stream_count) {
    Reset();
    stream_count = std::max(1, stream_count);
    streams_.reserve(static_cast<std::size_t>(stream_count));
    for (int i = 0; i < stream_count; ++i) {
        cudaStream_t stream = nullptr;
        const cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            EDGE_LOG_ERROR("cudaStreamCreateWithFlags failed for stream_index="
                           << i << ": " << cudaGetErrorString(status));
            Reset();
            return false;
        }
        streams_.push_back(stream);
    }
    next_.store(0);
    EDGE_LOG_INFO("CudaStreamPool initialized, stream_count=" << streams_.size());
    return true;
}

CudaStreamLease CudaStreamPool::Acquire() {
    if (streams_.empty()) {
        return {};
    }
    const int index = next_.fetch_add(1) % static_cast<int>(streams_.size());
    return CudaStreamLease(streams_[static_cast<std::size_t>(index)], index);
}

void CudaStreamPool::Reset() {
    for (cudaStream_t stream : streams_) {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
    streams_.clear();
    next_.store(0);
}

int CudaStreamPool::Size() const {
    return static_cast<int>(streams_.size());
}

std::string CudaStreamPool::Name() const {
    std::ostringstream oss;
    oss << "CudaStreamPool(size=" << streams_.size() << ")";
    return oss.str();
}

bool AllocatePinnedFloatTensor(
    const std::vector<int64_t>& shape,
    std::size_t elements,
    TensorBuffer& tensor) {
    if (elements == 0) {
        tensor.host_data.clear();
        tensor.ClearExternalHostData();
        tensor.shape = shape;
        return true;
    }

    void* raw = nullptr;
    const cudaError_t status = cudaHostAlloc(&raw, elements * sizeof(float), cudaHostAllocDefault);
    if (status != cudaSuccess) {
        EDGE_LOG_ERROR("cudaHostAlloc pinned float tensor failed: " << cudaGetErrorString(status));
        return false;
    }

    std::shared_ptr<float> pinned(
        static_cast<float*>(raw),
        [](float* ptr) {
            if (ptr != nullptr) {
                cudaFreeHost(ptr);
            }
        });
    tensor.shape = shape;
    tensor.SetExternalHostData(std::move(pinned), elements, true);
    return true;
}

}  // namespace edge

#endif  // defined(EDGE_ENABLE_CUDA) || defined(EDGE_ENABLE_TENSORRT_PLUGIN)
