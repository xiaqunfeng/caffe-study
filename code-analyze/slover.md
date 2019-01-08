## solver

Solver通过协调Net的前向推断计算和反向梯度计算(forward inference and backward gradients)，来对参数进行更新（即交替调用前向和反向算法来更新参数），从而达到减少loss的目的。实际上就是一种迭代的优化算法。

Caffe模型的学习被分为两个部分：

- 由Solver进行优化、更新参数
- 由Net计算出loss和gradient

Caffe提供了六种优化算法来求解最优参数，在solver配置文件中，通过设置type类型来选择。

- Stochastic Gradient Descent (`type: "SGD"`),
- AdaDelta (`type: "AdaDelta"`),
- Adaptive Gradient (`type: "AdaGrad"`),
- Adam (`type: "Adam"`),
- Nesterov’s Accelerated Gradient (`type: "Nesterov"`) and
- RMSprop (`type: "RMSProp"`)

Solver的流程：

- 1. 设计好需要优化的对象，以及用于学习的训练网络和用于评估的测试网络。（通过调用另外一个配置文件prototxt来进行）
- 2. 通过forward和backward迭代的进行优化来跟新参数。
- 3. 周期性地用测试网络评估模型性能（可设定多少次训练后，进行一次测试）
- 4. 在优化过程中记录模型和solver状态的快照（snapshot）

在每一次的迭代过程中，solver做了如下几步工作：

- 1、调用Net 的 forward算法来计算最终的输出值，以及对应的loss
- 2、调用Net 的 backward算法来计算每层的梯度（loss对每层的权重 w 和偏置 b 求导）
- 3、根据选用的slover方法，利用梯度进行参数更新
- 4、根据学习率(learning rate)，历史数据和求解方法更新solver的状态，使权重从初始化状态逐步更新到最终的学习到的状态。

Solvers的运行模式有CPU/GPU两种模式。

>以上摘自Caffe官方教程中译本_CaffeCN社区翻译，并加入少量修改。

## 配置文件

solver是caffe的核心模块，它协调着整个模型的运作。Solver定义了针对Net网络模型的求解方法，记录神经网络的训练过程，保存神经网络模型参数，中断并恢复网络的训练过程。自定义Solver能够实现不同的神经网络求解方式。

caffe程序运行必须的一个参数就是solver配置文件，该配置文件用来告知Caffe怎样对网络进行训练。示例：查看examples/mnist下的一个solver配置文件：lenet_solver.prototxt

```
net: "examples/mnist/lenet_train_test.prototxt"
test_iter: 100
test_interval: 500
base_lr: 0.01
momentum: 0.9
type: SGD
weight_decay: 0.0005
lr_policy: "inv"
gamma: 0.0001
power: 0.75
display: 100
max_iter: 20000
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
solver_mode: CPU
```

- net：设置深度网络模型。每一个模型就是一个net，需要在一个专门的配置文件中对net进行配置，每个net由许多的layer所组成
- test_iter：与test layer中的batch_size结合起来理解。假设mnist数据中测试样本总数为10000，batch_size为100，则需要迭代100次才能将10000个数据全部执行完。因此test_iter设置为100。执行完一次全部数据，称之为一个epoch
- test_interval：测试间隔。这里是每训练500次，才进行一次测试。
- base_lr：基础学习率
- momentum：上一次梯度更新的权重
- type：优化算法选择，默认是SGD，共有六种可选
- weight_decay：权重衰减项，防止过拟合的一个参数
- lr_policy：学习率的策略，有保持不变、sigmod衰减等等
- gamma：学习率相关，根据lr_policy计算是否需要而设
- power：学习率相关，根据lr_policy计算是否需要而设
- display：显示训练次数间隔。每训练100次，在屏幕上显示一次。如果设置为0，则不显示
- max_iter：最大迭代次数。太小，会导致没有收敛，精确度很低；太大，会导致震荡，难以收敛
- snapshot：快照。将训练出来的model和solver状态进行保存，snapshot用于设置训练多少次后进行保存，默认为0，不保存
- snapshot_prefix：设置snapshot保存路径
- solver_mode：设置运行模式。默认为GPU,如果你没有GPU,则需要改成CPU,否则会出错。

## slover类

