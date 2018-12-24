#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

// Blob可以支持的最高维数，目前设置为32
const int kMaxBlobAxes = 32;

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  // 默认构造函数: 初始化列表 {空函数体}
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  // 已废弃：通过设置数据维度（N,C,H,W）初始化
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  // 通过传入vector<int> 直接传入维数进行初始化
  explicit Blob(const vector<int>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  // 通过vector<int>参数设置shape_、count_和capacity_大小
  void Reshape(const vector<int>& shape);
  // 通过类BlobShape参数设置shape_、count_和capacity_大小
  void Reshape(const BlobShape& shape);
  // 通过其他已知的blob参数来设置shape_、count_和capacity_大小
  void ReshapeLike(const Blob& other);

  // 将blob的shape_和count_值转化为输出流string类型
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

  // 获得当前Blob的所有维度值shape_
  inline const vector<int>& shape() const { return shape_; }
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
  // 获得当前Blob的维数
  inline int num_axes() const { return shape_.size(); }
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

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
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

  // 完成梯度下降过程中的参数更新, 被网络中存储参数的Blob调用
  void Update();
  // Blob的数据持久化函数，通过Protobuf来做相应的序列化/反序列化操作
  void FromProto(const BlobProto& proto, bool reshape = true);
  void ToProto(BlobProto* proto, bool write_diff = false) const;

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

 protected:
  shared_ptr<SyncedMemory> data_;       // 存储前向传播的数据
  shared_ptr<SyncedMemory> diff_;       // 存储反向传播的导数、梯度、偏差
  shared_ptr<SyncedMemory> shape_data_; // 参数维度old version
  vector<int> shape_;                   // 参数维度, 若为4维，则依次为num、channels、height、width
  int count_;                           // Blob存储的元素个数(shape_所有元素乘积: n*c*h*w)
  int capacity_;                        // 当前Blob的元素个数（控制动态分配）, 因为blob会reshape

  DISABLE_COPY_AND_ASSIGN(Blob);        // 禁止使用Blob类的拷贝和赋值操作
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
