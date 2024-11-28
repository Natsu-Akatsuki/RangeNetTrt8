# The reason for why using FP16 can cause accuracy degradation

## Reason

As https://github.com/PRBonn/rangenet_lib/issues/9 points out, using FP16 can cause accuracy degradation based on original implementation. **The reason is that the ONNX model provided by the original implementation contains numerical instability.** (See https://github.com/pytorch/glow/issues/4654)

The instability stems from that the original onnx model (See https://www.ipb.uni-bonn.de/html/projects/semantic_suma/darknet53.tar.gz in [Here](https://github.com/PRBonn/rangenet_lib?tab=readme-ov-file#run-the-demo)) uses the `ReduceSum` operator, `Exp` operator, `Div` operator to realize `Softmax` operator. (You can use https://netron.app/ to visualize the onnx model) 

>  [!note]
>
> The exported onnx model did not directly contain the `Softmax` operator few years ago.

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/model.onnx.png" alt="model.onnx" style="zoom:67%;" />

## Solution

Solution 1: Do not use FP16, like in https://github.com/PRBonn/semantic_suma/pull/66, at the cost of incomplete model acceleration.

Solution 2: Set these three layers to use FP32, instead of applying FP16 suppression to all layers.

Solution 3 (Recommend):  Fold `ReduceSum` operator, `Exp` operator, `Div` operator into `Softmax` operator, like:


```plain
/// Fold Exp + ReduceSum + Div into Softmax
///    IN
///     |
///    Exp                IN
///   /   \                |
///  |  ReduceSum  -->  Softmax
///   \   /                |
///    Div                OUT
///     |
///    OUT
```

We provide a Python script (See script/fold_operator.py) to update the original onnx model. The new model can be found in [Here](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/releases/tag/v0.0.0-alpha).