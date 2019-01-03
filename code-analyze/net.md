## net

通过合成和自动微分，网络同时定义了一个函数和其对应的梯度。通过合成各层的输出来计算这个函数，来执行给定的任务，并通过合成各层的向后传播过程来计算来自损失函数的梯度，从而学习任务。Caffe模型是端到端的机器学习引擎。

Net是由一系列层组成的有向无环(DAG)计算图，Caffe保留了计算图中所有的中间值以确保前向和反向迭代的准确性。一个典型的Net开始于data layer------从磁盘中加载数据，终止于loss layer------计算分类和重构这些任务的目标函数。

Net由一系列层和它们之间的相互连接构成，用的是一种文本建模语言(protobuf)。Net是通过protobuf文件来描述整个Net是怎么由layer组成的。

前传(forward)过程为给定的待推断的输入计算输出。在前传过程中，Caffe组合每一层的计算以得到整个模型的计算”函数”。本过程自底向上进行。

反传(backward)过程根据损失来计算梯度从而进行学习。在反传过程中，Caffe通过自动求导并反向组合每一层的梯度来计算整个网络的梯度。这就是反传过程的本质。本过程自顶向下进行。

反传过程以损失开始，然后根据输出计算梯度。根据链式准则，逐层计算出模型其余部分的梯度。有参数的层，会在反传过程中根据参数计算梯度。

实现方法：

(1)、`Net::Forward()`和`Net::Backward()`方法实现网络的前传和反传，而`Layer::Forward()`和`Layer::Backward()`计算每一层的前传和后传。

(2)、每一层都有`forward_{cpu,gpu}()`和`backward_{cpu,gpu}`方法来适应不同的计算模式。由于条件限制或者为了使用便利，一个层可能仅实现了CPU或者GPU模式。

> 以上摘自Caffe官方教程中译本_CaffeCN社区翻译，并加入少量修改。

总结：

Net是Caffe代码中一个比较核心的类，它封装了所有的Layer，按照网络定义文件将需要的layers和中间blobs进行实例化，并将所有的Layers组合成一个有向无环图，构建起了整个神经网络。

Net还提供了在整个网络上进行前向传播与后向传播的接口，以及核心数据结构的访问结构，使得再上层的Solver可以利用Net比较轻松地实现Train和Test的策略。

## 重要成员

```
// layers_中存放着网络的所有layers，也就是Net类的实例保存着网络定义文件中所有layer的实例
vector<shared_ptr<Layer<Dtype> > > layers_; 

// blobs_中保存着网络所有的中间结果，即所有layer的输入数据（bottom blob）和输出数据（top blob）
vector<shared_ptr<Blob<Dtype> > > blobs_; 

// bottom_vecs_保存的是各个layer的bottom blob的指针，这些指针指向blobs_中的blob。
// bottom_ves.size()与网络layer的数量相等，由于layer可能有多个bottom blob
// 所以使用vector<Blob<Dtype>*>来存放layer-wise的bottom blob。top_vecs_同理
vector<vector<Blob<Dtype>*> > bottom_vecs_; 
vector<vector<Blob<Dtype>*> > top_vecs_; 

// 存放的是指向网络参数的指针
// 直接拥有参数的是layer，params_保存的只是网络中各个layer的参数的指针
// learnable_params_保存的是各个layer中可以被学习的参数
vector<shared_ptr<Blob<Dtype> > > params_; 
vector<Blob<Dtype>*> learnable_params_; 
```

## 成员函数

### init函数

初始化函数重要的步骤和函数如下：

```
1.     FilterNet(in_param,&filtered_param);
此函数的作用就是模型参数文件（*.prototxt）中的不符合规则的层去掉。例如：在caffe的examples/mnist中的lenet网络中，如果只是用于网络的前向，则需要将包含train的数据层去掉。

2、InsertSplits(filtered_param,&param);
此函数作用是，对于底层一个输出blob对应多个上层的情况，则要在加入分裂层，形成新的网络。这么做的主要原因是多个层反传给该blob的梯度需要累加。

例如：LeNet网络中的数据层的top label blob对应两个输入层，分别是accuracy层和loss层，那么需要在数据层在插入一层。

3、layers_.push_back();
该行代码是把当前层的参数转换为shared_ptr<Layer<Dtype>>，创建一个具体的层，并压入到layers_中

4、AppendBottom();
此函数为该层创建bottom blob，由于网络是堆叠而成，即：当前层的输出 bottom是前一层的输出top blob，因此此函数并没没有真正的创建blob，只是在将前一层的指针压入到了bottom_vecs_中。

5、AppendTop();
此函数为该层创建top blob，该函数真正的new的一个blob的对象。并将topblob 的指针压入到top_vecs_中

 6、layers_[layer_id]->SetUp();
  前面创建了具体的层，并为层创建了输入bottom blob 和输出top blob。改行代码这是启动该层，setup()函数的功能是为创建的blob分配数据内存空间，如有必要还需要调整该层的输入bottom blob 和输出top blob的shape。

 7、AppendParam();
 对于某些有参数的层，例如：卷基层、全连接层有weight和bias。该函数主要是修改和参数有关的变量，实际的层参数的blob在上面提到的setup()函数中已经创建。如：将层参数blob的指针压入到params_。
```

代码注释（init函数非常长）：

```
template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Set phase from the state. 训练网络还是测试网络
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  // 传入网络结构参数，根据LayerParameter中的include和exclude来确定该层是否应该包含在网络中
  // 返回过滤过后的网络参数
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  // 对过滤后的参数拷贝一个副本
  // 如果底层的输出(top)blob对应多个输入的时候，在该层增加分裂层SplitLayer，形成新的网络
  NetParameter param;
  InsertSplits(filtered_param, &param);

  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  // 参数初始化, 使用resize改变容器大小，并使用默认构造函数创建对象
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  // 大循环，处理每一层
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Inherit phase from net if unset.
    // 如果当前层没有设置phase，则当前层的phase从网络net的phase中继承
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    // 层结构参数
    const LayerParameter& layer_param = param.layer(layer_id);
    // 是否设置了对输入求导,参考caffe.proto里LayerParameter的propagate_down参数
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    // 创建具体的layer，压入layer_中
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    // 将创建的layer名压入layer_names_中
    layer_names_.push_back(layer_param.name());
    LOG_IF(INFO, Caffe::root_solver())
      << "Creating Layer " << layer_param.name();
    // 判断该层是否需要反向传播
    bool need_backward = false;

    // Figure out this layer's input and output
    // 计算每一层的输入输出
    // 注：第一层是数据输入层，没有输入bottom，所以会跳过该循环
    // net中bottom/top是交替初始化的,前一层的top是后一层的bottom
    // 前一层top的available_blobs/blob_name_to_idx参数就是后一层的bottom参数
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    // 每一层输出数据的个数
    int num_top = layer_param.top_size();
    // 对该层的每个输出数据循环
    for (int top_id = 0; top_id < num_top; ++top_id) {
      //通过AppendTop和AppendBottom, bottom_vecs_和top_vecs_连接在了一起
      //在AppendTop中会往available_blobs添加某层的输出blob,在AppendBottom中会
      //从available_blobs中删除前一层的输出blob，所有layers遍历完后剩下的就
      //是整个net的输出blob
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    // 如果AutoTopBlobs为true，且层参数中指定的输出blob数比需要的少，补上
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    // 启动，调用layer类的setup函数初始化
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    // 对每一层的输出blob循环
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      // 计算所需要的内存
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);

    // LayerParameter中定义的参数个数
    const int param_size = layer_param.param_size();
    // 该层实际参数个数
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    // 循环每个可学习的参数
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      // 如果prototxt文件没有设置param，则使用默认param
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      // 学习率(lr_mult)不等于0，表示需要对这个可学习的参数反向求导
      const bool param_need_backward = param_spec->lr_mult() != 0;
      // need_backward是当前层是否要做反向传播计算的最终判断:
      // need_backward由所有blob_need_backward_和param_need_backward_组合得到
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    // 循环该层的每个可学习参数
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      // 设置该层权值的一些参数，学习率，正则率，参数id等
      // param：整个网络参数，layer_id:层数id，param_id:可学习参数id
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    // 最后设置反向传播标志
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  } // 大的层循环结束
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip backward
  // computation for the entire layer
  // 寻找反向传播过程中哪些blobs对最终的loss有影响
  // 如果某个blob对最终的loss没有贡献，则跳过该blob的求梯度计算
  // 检查是否所有的blobs都不需要求梯度，如果是，可以对整个layer不用BP计算
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  // 层循环，从后向前，将不需要计算backward的层和blob标记出来
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    // 遍历该层每个top blob，确定该层是否输出loss，是否需要进行backward计算
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      // 如果当前层是最终输出层，或者当前top blob为最终loss做出贡献，layer_contributes_loss置true
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      // 只要该层有一个blob对loss有贡献，说明该层对loss有贡献, 退出循环
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    // 如果这一层跳过梯度计算，那么这一层所有的输入blobs都不需要计算梯度
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      // 如果该层为loss做出贡献，将做出贡献的blob插入blobs_under_loss中去
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        // 如果该层没有为loss做出贡献, 该层就不需要backward计算
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  } // 从后向前的层循环结束

  // Handle force_backward if needed.
  // 强制计算梯度
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  // 最后，输入输出blob中除了输入blob剩下的都作为网络的输出
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}
```

其他代码注释详见文件