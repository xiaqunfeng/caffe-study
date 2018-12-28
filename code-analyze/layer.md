## layer

## 关于layer

Layer是Caffe模型的本质内容和执行计算的基本单元。Layer可以进行很多运算，如convolve(卷积)、pool(池化)、inner product(内积)，rectified-linear和sigmoid等非线性运算，元素级的数据变换，normalize(归一化)、load data(数据加载)、softmax和hinge等losses(损失计算)。可在Caffe的[层目录](http://caffe.berkeleyvision.org/tutorial/layers.html )中查看所有操作，其囊括了绝大部分目前最前沿的深度学习任务所需要的层类型。

一个layer通过bottom(底部) 连接层接收blobs数据，通过top(顶部)连接层输出blobs数据。Caffe中每种类型layer的参数说明定义在caffe.proto文件中，具体的layer参数值则定义在具体应用的prototxt网络结构说明文件中。

在Caffe中，一个网络的大部分功能都是以layer的形式去展开的。在创建一个Caffe模型的时候，也是以layer为基础进行的，需按照caffe.proto中定义的网络及参数格式定义网络prototxt文件。在.prototxt文件中会有很多个layer {  } 字段。

每一个layer都定义了3种重要的运算：setup(初始化设置)，forward(前向传播)，backward(反向传播)。

- (1)、setup：在模型初始化时重置layers及其相互之间的连接；
- (2)、forward：从bottom层中接收数据，进行计算后将输出送人到top层中；
- (3)、backward：给定相对于top层输出的梯度，计算其相对于输入的梯度，并传递到bottom层。一个有参数的layer需要计算相对于各个参数的梯度值并存储在内部。

特别地，forward和backward函数分别有CPU和GPU两张实现方式。如果没有实现GPU版本，那么layer将转向作为备用选项的CPU方式。这样会增加额外的数据传送成本(输入数据由GPU上复制到CPU，之后输出数据从CPU又复制回到GPU)。

总的来说，Layer承担了网络的两个核心操作：

- forward pass(前向传播)----接收输入并计算输出；
- backward pass(反向传播)----接收关于输出的梯度，计算相对于参数和输入的梯度并反向传播给在它前面的层。

由此组成了每个layer的前向和反向传播。

Layer是网络的基本单元，由此派生出了各种层类。在Layer中input data用bottom表示，output data用top表示。由于Caffe网络的组合性和其代码的模块化，自定义layer是很容易的。只要定义好layer的setup(初始化设置)、forward(前向传播，根据input计算output)和backward(反向传播，根据output计算input的梯度)，就可将layer纳入到网络中。

> 以上摘自Caffe官方教程中译本_CaffeCN社区翻译，并加入少量修改。

## message信息

caffe.proto中，有2个message与blob有关，定义如下：

Phase

```
enum Phase {
   TRAIN = 0;
   TEST = 1;
}
```

LayerParameter

```
message LayerParameter {
  optional string name = 1; // layer名字，可自定义
  optional string type = 2; // layer类型，在具体的layer中写定，可通过type()函数获取之
  repeated string bottom = 3; // 输入bottom blob的名字，可以有多个
  repeated string top = 4; // 输出top blob的名字，可以有多个

  optional Phase phase = 10; // 计算的阶段，训练还是测试

  repeated float loss_weight = 5; // 和top blob数量相同，每层分配一个默认值，通常为0或1
  repeated ParamSpec param = 6; // train时用到的参数
  repeated BlobProto blobs = 7; // 含有数字参数层的blob
  repeated bool propagate_down = 11; // 是否BP到每个bottom。数量为0或者等于bottom的个数

  // 控制是否以及何时在网络中包含一个layer的规则
  repeated NetStateRule include = 8;
  repeated NetStateRule exclude = 9;

  optional TransformationParameter transform_param = 100; // 数据预处理参数
  optional LossParameter loss_param = 101; // loss层共享的参数

  // 具体layer的参数
  ...
  optional BatchNormParameter batch_norm_param = 139;
  optional BiasParameter bias_param = 141;
  optional ConvolutionParameter convolution_param = 106;
  ...
}
```

## layer类成员变量

类Layer：抽象基类，有纯虚函数，不能实例化，定义了所有layer的基本接口，具体的每个layer完成一类特定的计算

```
 protected:
   LayerParameter layer_param_;  
   Phase phase_;         
   vector<shared_ptr<Blob<Dtype> > > blobs_;  
   vector<bool> param_propagate_down_;       
   vector<Dtype> loss_;                     
```

- layer_param_： protobuf文件中存储的layer参数，具体参数详见caffe.proto中的message LayerParameter
- phase_ ：layer状态：指定参与网络的是train还是test
- blobs_：用于存储layer的学习的参数如权值和偏置，使用向量是因为权值参数和偏置分开保存在两个blob中
- param_propagate_down_：标志每个可学习参数blob是否需要计算反向传递的梯度值
- loss_：非LossLayer为零，LossLayer中表示每个top blob计算的loss的权重

## layer类成员函数

### 初始化

```
  // bottom层的输入数据，blob中的存储空间已申请
  // top层输出数据，blob对象已构造，但是其存储空间未申请
  // 具体空间大小需根据bottom blob大小和layer_param_共同决定，具体在Reshape函数现实
  // 函数功能：实现公共layer的setup功能，此方法非虚函数, 所以不需要重写
  // 1. 检查blobs输入(bottom)和输出(top)个数是否正确，每层处理的输入输出数据不一样
  // 2. 调用LayerSetUp来为各个类型的layer执行特定的setup, 各子类需要重写该函数完成初始化
  // 3. Reshape: 设置blobs输出（top）及内部缓存的大小, 即为top blob分配合适大小的存储空间
  // 4. 为每个损失权重为非零的top blob设置损失权重乘子
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }
```

初始化的四个函数，除了SetLossWeights是定义在类内的内联函数外，其他三个均为虚函数。

```
  // 实现指定层的初始化，包括从layer_param_读入并处理相关的层权值和偏置参数
  // 调用Reshape函数申请top blob的存储空间
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  
  // 纯虚函数，每个子类Layer必须重写的Reshape函数，完成top blob形状的设置并为其分配存储空间
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
```

### 前向传播和反向传播

```
  // inline函数，调用Forward_cpu和Forward_gpu这两个虚函数来完成数据前向传递
  // 根据执行环境的不同每个子类Layer必须重写CPU和GPU版本
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
    // 用法类似Forward
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);
```

这两个函数都在类内声明，类外实现。CPU的前向/反向传播为纯虚函数，由子类实现：

```
 protected:
   // CPU实现layer的前向传播
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
      
  // GPU实现layer的前向传播
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
      
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }
```

### blob相关

下面几个函数主要设置bottom或者top blob的数量状态，通常需要layer类的派生类重写，因为不同层指定的输入输出数量不同

```
  // 获得layer所需的bottom blobs的个数
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  // 获得layer所需的bottom blobs的最少个数
  virtual inline int MinBottomBlobs() const { return -1; }
  // 获得layer所需的bottom blobs的最多个数
  virtual inline int MaxBottomBlobs() const { return -1; }
  
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return -1; }
  virtual inline int MaxTopBlobs() const { return -1; }
  
  // 判断layer所需的bottom blobs和top blobs的个数是否相等
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }
  
  // 判断layer所需的的top blobs是否需要由Net::Init来创建
  virtual inline bool AutoTopBlobs() const { return false; }
  
  // 判断layer指定的bottom blob是否需要强制梯度返回
  // 有些layer不需要梯度信息
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
```

> 其他函数详见 hpp文件。

## 内联函数

### Forward

```
// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
// 前向传播，通过输入bottom blobs，计算输出top blobs和loss值
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  return loss;
}
```

### Backward

```
// 反向传播，通过给定top blob误差梯度，计算bottom blob误差梯度
template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}
```

### ToProto

```
// Serialize LayerParameter to protocol buffer
// 序列化函数，将layer参数写入protobuf文件
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}
```

# Layer.cpp

```
// 模板显式实例化
INSTANTIATE_CLASS(Layer);
```

在common.hpp中定义

```
// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>
```

