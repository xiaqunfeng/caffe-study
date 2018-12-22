#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// 默认构造函数
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

// 带参数的构造函数，注意是explicit类型，只能直接初始化, 不能复制初始化
SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

// 析构函数，检查内部自己申请的空间自己释放
SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

// 将最新数据复制到cpu上
inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {                  // 检查最新数据的位置
  case UNINITIALIZED:               // 未初始化：分配内存，初始化数据
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;            // 将最新数据的位置置为在cpu侧
    own_cpu_data_ = true;           // 内部自己分配的空间，置为true
    break;
  case HEAD_AT_GPU:                 // 最新数据在GPU侧
#ifndef CPU_ONLY                    // 将数据从gpu复制到cpu，如果cpu侧指针为NULL，则分配空间
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;                 // 设置最新数据标志为共享, 表示cpu和gpu数据一致
#else
    NO_GPU;                         // 如果使用CPU_ONLY模式，什么都不做，打印一条信息
#endif
    break;
  case HEAD_AT_CPU:                 // 最新数据本来就在CPU侧，什么都不做
  case SYNCED:                      // 最新数据是共享的，什么都不做
    break;
  }
}

// 将最新数据复制到gpu上
// 代码逻辑同 to_cpu()，不再逐行注释
inline void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

// 访问cpu上数据，先将最新代码复制到cpu上，再返回只读指针
const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

// 将cpu_ptr_ 指针指向外部数据 
void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {      // 如果cpu自己分配过内存，先释放
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;          // 指向外部数据
  head_ = HEAD_AT_CPU;      // CPU侧更新了数据，所以最新数据在cpu侧
  own_cpu_data_ = false;    // 表示数据来源于外部，不是内部分配内存
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

// 访问cpu上数据，先将最新代码复制到cpu上, 再返回可写指针
// 不管之前状态是哪种，此时将状态置为在cpu上, 认为CPU侧是最新的，意味着调用者可以修改CPU侧数据
void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
// 以流的方式将数据同步到gpu上
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);      // 检查数据位于cpu上
  if (gpu_ptr_ == NULL) {           // 在gpu上分配数据的存储空间
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;  
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream)); // 将cpu上数据同步到gpu
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;                   // 将最新数据指针置为共享
}
#endif

// 校验当前gpu设备
void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  // 校验gpu_ptr所指向的设备是否是构造时获取的gpu设备
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe

