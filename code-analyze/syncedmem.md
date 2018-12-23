## SyncedMemory类

该类用来管理caffe中Blob的内存，所以在看blob代码之前先看一下该类的实现。内存管理的方法代码都封装在syncedmem.hpp与syncedmem.cpp两个文件中，两个文件代码都比较短，读起来比较方便。

`SyncedMemory`的两个优点：

- 屏蔽了CPU和GPU上的内存管理以及数据同步细节
- 使用lazy的内存分配方式（通过`enum SyncedHead`状态控制来实现），在数据访问时才分配，而不是立马分配，提高效率以及节省内存

### 类的数据成员

```
class SyncedMemory {
...
   private:
        ...
        void* cpu_ptr_;
        void* gpu_ptr_;
        size_t size_;  
        SyncedHead head_;
        bool own_cpu_data_;
        bool cpu_malloc_use_cuda_;
        bool own_gpu_data_;
        int device_;
        ...
};

```

1、其中 head_ 指向目前最新数据块的位置，即最后更新过数据的位置，用于cpu和gpu数据同步。这是一个枚举类型：

```
enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
```

依次表示：数据未初始化、最新数据在cpu上、最新数据在gpu上、最新数据cpu和gpu共享。

2、size_ 表示数据所占内存大小

3、cpu_ptr_ 和 gpu_ptr_ 分别表示指向cpu和gpu侧的指针，通过该指针访问数据

这两个指针指向的数据空间有两种来源，一种是对象自己内部分配的，一种是外部指定的

4、own_cpu_data_ 和 own_gpu_data_ 表示对象是否是自己内部分配的

- own_cpu_data_ 为true时，表示cpu_ptr_是对象内部调用CaffeMallocHost分配的CPU内存
- own_gpu_data_ 为true时，表示gpu_ptr_是对象内部调用cudaMalloc分配的GPU内存

这两个标志位在 set_cpu/gpu_data() 及函数析构的时候会用于判断是否需要释放当前的数据，如果是自己内部申请的，就要内部释放。

5、cpu_malloc_use_cuda_ 为true时，使用cudaMallocHost分配页锁定内存，否则使用系统malloc分配可分页内存

6、device_ 表示gpu的设备编号

### 类的成员函数

```
class SyncedMemory {
 public:
  ...
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  ...

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();
  ...
};
```

1、`cpu_data()` 函数用于获取cpu数据的const指针，只读不写

cpu_data()函数的核心是to_cpu()函数（在syncedmem.cpp中定义）：

```
inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}
```

这个函数的作用是检查最新数据的位置（即查看 head_ 的标志）

- 数据未初始化 —— 初始化，并将最新数据标志置为位于cpu
- 最新数据在gpu上 —— 如果指定了CPU_ONLY，则输出NO_GPU并什么都不做；否则将数据复制到cpu，并将数据标志置为共享
- 最新数据在cpu上或者是共享的 —— 什么都不做

2、`mutable_cpu_data() ` 返回可写的cpu指针

它与 cpu_data() 的区别是：在执行 to_cpu() 后，不论 head_ 当前状态如何，将最新数据的位置置为位于cpu上，认为cpu侧数据是最新的，所以调用者可以修改cpu侧数据。

```
void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}
const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}
```

3、`set_cpu_data(void* data)`

设置cpu数据，即将cpu_ptr_ 指针指向外部数据。

4、`gpu_data(), mutable_gpu_data(), set_gpu_data(void* data) ` 功能同cpu

5、`async_gpu_push(const cudaStream_t& stream)`

功能就是以流的形式将数据同步到gpu上。具体操作就是在gpu上分配存储空间，将cpu的数据同步至gpu，并将head_ 置为共享。

总结一下CPU/GPU数据同步的功能：

- 第一次访问某一侧数据时分配该侧内存，如果不曾访问过则不分配内存，按需分配来节省内存。
- 用`head_`来指示最近一次数据更新发生在哪一侧，仅在调用另一侧数据时才将该侧数据同步过去，如果访问的仍是该侧，则不会发生同步。
- 当两侧已同步都是最新时，即`head_=SYNCED`，访问任何一侧都不会发生数据同步。

> 其他成员函数的实现代码详见syncedmem-annotation.cpp

### 同步示例

Caffe官网上提供的何时发生内存同步的例子：

```
// Assuming that data are on the CPU initially, and we have a blob.
const Dtype* foo;
Dtype* bar;
foo = blob.gpu_data(); // data copied cpu->gpu.
foo = blob.cpu_data(); // no data copied since both have up-to-date contents.
bar = blob.mutable_gpu_data(); // no data copied.
// ... some operations ...
bar = blob.mutable_gpu_data(); // no data copied when we are still on GPU.
foo = blob.cpu_data(); // data copied gpu->cpu, since the gpu side has modified the data
foo = blob.gpu_data(); // no data copied since both have up-to-date contents
bar = blob.mutable_cpu_data(); // still no data copied.
bar = blob.mutable_gpu_data(); // data copied cpu->gpu.
bar = blob.mutable_cpu_data(); // data copied gpu->cpu.
```

> 建议不修改数据时调用`const`函数，不要调用`mutable`函数

## 内存分配与释放

在 syncedmem.hpp 中有两个内联函数，不属于 SyncedMemory 类：CaffeMallocHost，CaffeFreeHost，用于内存的申请和释放。

代码逻辑比较简单，如果是CPU模式，那么调用malloc和free（或者intel的mkl模块）来申请/释放内存，否则调用CUDA的 cudaMallocHost 和 cudaFreeHost 来申请/释放显存。

```
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}
```

### 关于 Pinned 和 Non-Pinned Memory

当需要分配CPU内存且会传输到GPU时，有两种方式可以选择：`pinned` 和 `non-pinned`。

Pinned memory使用`cudaMallocHost`来分配CPU内存，可以防止内存页被交换出去，因此可以提供更高的传输速度。Non-pinned memory使用`malloc`函数分配CPU内存。但是Pinned memory 比Non-pinned memory有更昂贵的内存分配和释放，因为`cudaMallocHost`分配和释放CPU内存相比`malloc`更加耗时。

结论：

- 使用`Pinned Memory`方式的CPU和GPU之间传输速度更大，但是分配和释放的耗时更大
- 使用`Non-Pinned Memory`的方式CPU和GPU之间传输耗时更大，但是分配和释放更快
- 当传输的数据比较大，且均需从CPU传至GPU和从GPU传至CPU时，使用`Pinned Memory`可以获得更好的性能