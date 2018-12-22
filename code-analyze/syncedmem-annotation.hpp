#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
// 如果CUDA可用并处于GPU模式，则使用 cudaMallocHost 为主机分配固定内存。它避免了动态固定传输(DMA)。
// 在单GPU案例中，性能提升微乎其微，但在并行训练中可能提升更显著。
// 更重要的是，它提升了在多GPU上大模型的稳定性。
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL // 使用intel的mkl工具
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();           // 返回cpu上数据的只读指针
  void set_cpu_data(void* data);    // 设置cpu数据：将cpu_ptr_ 指针指向外部数据
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();         // 返回cpu上数据的可写指针
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED }; // 数据指针状态
  SyncedHead head() const { return head_; }     // 返回最新数据的指针位置
  size_t size() const { return size_; }         // 返回数据的大小

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);  // 以流的形式将cpu数据以异步的方式同步到gpu
#endif

 private:
  void check_device();

  void to_cpu();                // 数据复制到cpu
  void to_gpu();                
  void* cpu_ptr_;               // cpu侧数据指针
  void* gpu_ptr_;
  size_t size_;                 // 数据所占内存大小
  SyncedHead head_;             // 指示最新数据在哪一侧，调用另一侧数据时需要将该侧数据同步过去
  bool own_cpu_data_;           // cpu_ptr_是否为对象内部调用CaffeMallocHost分配的CPU内存
  bool cpu_malloc_use_cuda_;    // 指示是否使用cudaMallocHost分配页锁定内存
  bool own_gpu_data_;           // gpu_ptr_是否为对象内部调用cudaMalloc分配的GPU内存
  int device_;                  // GPU设备号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
