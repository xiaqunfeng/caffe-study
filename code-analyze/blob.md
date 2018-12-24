## Blob

Blob 是 caffe 中处理和传递实际数据的数据封装包，其可根据CPU主机到GPU设备的同步需要，屏蔽CPU/GPU混合运算在计算上的开销，主机和设备上的内存按需分配，以提高内存使用效率（SyncedMemory内存管理类）。

caffe基于blob存储和交换数据，为便于优化，blob提供统一的内存接口来存储某种类型的数据，例如批量图像数据、模型参数以及用来进行优化的导数。

Blob数据可以通过Protobuf来做相应的序列化操作，ToProto和FromProto两个函数完成相应的序列化、反序列化(数据解析)操作。

### blob数据存储

从数学意义上说，blob是按C风格连续存储的N维数组，即在内部所存储的数据是一块连续的内存，实际上是一个一维的指针。

对于批量图像数据来说，blob常规的维数为：图像数量N、通道数K、图像高度H、图像宽度W。blob按行为主(row-major)进行存储，所以一个4维blob中，坐标为`(n,k,h,w)`的实际物理位置为`((n*K+k)*H+h)*W+w`，最右边的维度更新最快。

- Number/N是每个批次处理的数据量。批量处理信息有利于提供设备处理和交换的数据的吞吐率。在ImageNet上每个训练批量为256张图像，则N=256；
- Channel/K是特征维度，例如对RGB图像来说，可以理解为通道数量，K=3；如果是网络中间结果，就是feature map的数量；
- H、W：如果是图像数据，可以理解为图像的高度和宽度；如果是参数数据，可以理解为滤波核的高度和宽度。

虽然Caffe的图像应用例子中很多blobs都是4维坐标，但是对于非图像应用任务，blobs也完全可以照常使用。例如，如果你仅仅需要类似于传统多层感知机那样的全连接层，使用2维的blobs（形式为(N,D)），之后再调用 InnerProductLayer。

参数Blob的维度是根据层的类型和配置而变化的。一个卷积层中若有96个空间维度为 `11*11`、输入为3通道的滤波器，那么其blob维度是 `96*3*11*11`。对于一个输入是 1024 维（输入通道数），输出是 1000 维（输出通道数）的内积层/全连接层，参数blob维度是 `1000*1024`。

> 以上摘自Caffe官方教程中译本_CaffeCN社区翻译，并加入少量修改。

### message信息

caffe.proto中，有3个message与blob有关，定义如下：

1、BlobShape

```
// Specifies the shape (dimensions) of a Blob.
// 指定数据块Blob的维度，若为4维，则为num、channel、height、width
message BlobShape {
  repeated int64 dim = 1 [packed = true];
}
```

2、BlobProto

```
message BlobProto {
  optional BlobShape shape = 7;                     // BlobShappe类对象
  repeated float data = 5 [packed = true];          // 前向传播的data，float类型
  repeated float diff = 6 [packed = true];          // 反向传播的diff，float类型
  repeated double double_data = 8 [packed = true];  // 前向传播的data，double类型
  repeated double double_diff = 9 [packed = true];  // 反向传播的diff，double类型

  // 4D dimensions -- deprecated.  Use "shape" instead.
  // 已废弃，使用BlobShape shape替代
  optional int32 num = 1 [default = 0];
  optional int32 channels = 2 [default = 0];
  optional int32 height = 3 [default = 0];
  optional int32 width = 4 [default = 0];
}
```

3、BlobProtoVector

```
// The BlobProtoVector is simply a way to pass multiple blobproto instances
// around.
// 存放多个BlobProto实例
message BlobProtoVector {
  repeated BlobProto blobs = 1;
}
```

### shared_ptr

`std::shared_ptr` 是通过指针保持对象共享所有权的智能指针。多个 `shared_ptr` 对象可占有同一对象。当最后一个占有对象的`shared_ptr`被销毁或被赋值为另一指针时，对象会被自动销毁并释放内存（详见[shared_ptr](https://zh.cppreference.com/w/cpp/memory/shared_ptr)）。

### blob类数据成员

```
class Blob {
...
 protected:
  shared_ptr<SyncedMemory> data_;        // 存储前向传递数据
  shared_ptr<SyncedMemory> diff_;        // 存储反向传递梯度
  shared_ptr<SyncedMemory> shape_data_;  // 参数维度old version
  vector<int> shape_;                    // 参数维度
  int count_;                            // Blob存储的元素个数（shape_所有元素乘积: n*k*h*w）
  int capacity_;                         // 当前Blob的元素个数（控制动态分配）
...
}
```

实际是在`SyncedMemory`上做了一层封装。

### blob类成员函数

> 主要介绍各函数的功能，部分函数的实现未完全展开，详见 blob.cpp

**1、reshape**

Reshape函数对Blob的形状进行初始化或改变。

> 在网络传播中，因为每层数据的维度长宽等都不一样，layer中也会调用Reshape函数来实现数据维度的对齐。

```
  // 通过vector<int>参数设置shape_、count_和capacity_大小
  void Reshape(const vector<int>& shape);
  // 通过类BlobShape参数设置shape_、count_和capacity_大小
  void Reshape(const BlobShape& shape);
  // 通过外部的blob参数来设置shape_、count_和capacity_大小
  void ReshapeLike(const Blob& other);
```

**2、shape**

```
  // 获得当前Blob的所有维度值shape_
  inline const vector<int>& shape() const { return shape_; }
  
  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  // 对输入的Blob维度索引值进行判断，返回有效的索引值，支持负数输入
  // *  0 <= index < num_axes(), 直接返回
  // * -num_axes <= index <= -1, 返回num_axes() - (-index)
  // * 其他情况报错
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }
  
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  // 获得当前Blob指定索引的维度值
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  
  // 获得当前Blob指定索引的维度值, 先进行判断，再调用shape()函数来完成
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }
```

**3、count**

```
  // 获得当前Blob的元素个数,即shape_结构中各维度值的乘积：N*C*H*W
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  // 根据指定的start_axis和end_axis计算blob元素个数, 即shape_中指定维度值的乘积
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  // 根据指定的start_axis计算blob元素个数，即shape_中从指定维度开始到末尾各维度值的乘积
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }
```

**4、offset**

```
  // 根据num、channels、height、width计算在数组中的偏移量,计算公式:((n*C+c)*H+h)*W+w
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  // 根据vector<int> index计算偏移量：((n*C+c)*H+h)*W+w
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
```

**5、cpu和gpu内存数据访问**

```
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  // 从source blob中拷贝数据到当前blob
  // 如果copy_diff为false，拷贝data_数据，否则，拷贝diff_数据
  // 如果reshape为false，要求source blob预先shape成与当前blob一致，否则，根据二者shape是否一致决定是否执行reshape操作
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);
      
  // 根据指定的偏移量获得前向传播数据data_的一个元素的值
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  // 根据指定的偏移量获得反向传播梯度diff_的一个元素的值
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  // 同上面的data_at，只不过输入是vector矢量
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  // 同上面的diff_at，只不过输入是vector矢量
  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  // 获取前向传播数据 data_ 的指针
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  // 获取反向传播梯度 diff_ 的指针
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }
```

**6、SyncedMemory封装**

```
  // Blob的CPU和GPU数据访问函数，调用SyncedMemory内存管理类中同名函数来实现
  // mutable_ 前缀得到Blob数据的可写指针，const函数得到只读指针
  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;
  void set_gpu_data(Dtype* data);
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
```

**7、update**

```
// 完成梯度下降过程中的参数更新, 被网络中存储参数的Blob调用
// 调用caffe_axpy函数重新计算data_(weight，bias 等减去对应的导数): data_ = -1 * diff_ + data_
// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}
```

其中caffe_axpy函数：

```
template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }
```

功能： Y=alpha*X+Y 
N：为X和Y中element的个数

**8、数据持久化**

Blob的数据持久化函数，通过Protobuf来做相应的序列化/反序列化操作

```
 // 将BlobProto的shape/data/diff分别copy给当前blob的shape_/data_/diff_完成数据解析(反序列化)，若reshape参数为true，则会对当前的blob重新进行reshape
  void FromProto(const BlobProto& proto, bool reshape = true);
  
  // 将Blob的shape_/data_/diff_(如果write_diff为true)分别copy给BlobProto的shape/data/diff完成序列化
  void ToProto(BlobProto* proto, bool write_diff = false) const;
```

**9、工具函数**

```
  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  // 计算data_的L1范式：向量中各个元素绝对值之和
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  // 计算diff_的L1范式：向量中各个元素绝对值之和
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  // 计算data_的L2范式平方：向量中各元素的平方和
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  // 计算diff_的L2范式平方：向量中各元素的平方和
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  // 将data_数据按照常数因子缩放
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  // 将diff_数据按照常数因子缩放
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  // 将外部指定的blob的data_指针指向给当前blob的data_,以实现共享data_
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  // 将外部指定的blob的diff_指针指向给当前blob的diff_,以实现共享diff_
  void ShareDiff(const Blob& other);

  // 比较两个blob的shape是否相同
  bool ShapeEquals(const BlobProto& other);
```

