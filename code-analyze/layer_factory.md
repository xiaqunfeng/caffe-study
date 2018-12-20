## layer_factory

caffe层的设计用到了设计模式里[工厂模式](https://blog.csdn.net/wuzhekai1985/article/details/6660462)中的工厂方法模式。

### LayerRegistry类

主要的实现是由该类来完成的。内部维护了一个map，key为layer的string（类型名称），value为对应Creator（工厂函数），caffe就是通过该map来管理string和Creator映射关系的。

```
typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
typedef std::map<string, Creator> CreatorRegistry;
```

map类型为`CreatorRegistry`，实际类型为`std::map<string, Creator>`。

### layer的注册

caffe通过两组宏来实现layer的注册：

1、先来看宏 `REGISTER_LAYER_CLASS`

```
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)`
```

该宏实现为指定type的Layer创建一个Creator工厂函数，然后调用`REGISTER_LAYER_CREATOR`将工厂函数和Layer的类型名进行注册，支持两种Layer的数据类型，float和double。

在每个XX_layer.cpp文件末尾处调用，进行layer的添加注册，比如在`bias_layer.cpp`的末尾：

```
REGISTER_LAYER_CLASS(Bias);
```

2、再来看宏 `REGISTER_LAYER_CREATOR`

```
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \
```

声明了两个变量，分别对应float和double两种类型，两个变量通过调用LayerRegisterer类的构造函数来完成初始化

```
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};
```

### LayerRegisterer类

代码如上，该类只有一个方法，即构造函数，在LayerRegisterer的构造方法中调用了`LayerRegistry`类的静态方法`AddCreator`，将新的layer和creator注册并添加到registry list中去。

```
class LayerRegistry {
...
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }
  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }
...
};
```

以上实现了动态注册

### 创建layer

注册完后，在xxnet.prototxt中定义相关的layer参数，比如mnist文件夹下的lenet.prototxt中定义type为Pooling的layer：

```
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
```

在net.cpp中创建方式如下：

```
void Net<Dtype>::Init(const NetParameter& in_param) {
...
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
...
}
```

调用`LayerRegistry`类的静态方法`CreateLayer`来实现layer的创建

```
class LayerRegistry {
  ...
  // Get a layer using a LayerParameter.
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }
  ...
};
```

