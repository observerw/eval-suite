# 从头实现一个 Benchmark，以衡量推理轨迹长度平均值任务为例

[代码](../examples/thinking_token.py)

在之前我们实现了一些基于预定义评估方法的 Benchmark，如 Pass@k 等。这些 Benchmark 内部已经包含了许多逻辑，因此实现起来还算简单。现在让我们来从头实现一个新的 Benchmark，以此来展示如何使用 `Benchmark` 类来实现一个新的评估方法。

在本教程中，我们将“统计推理轨迹平均长度”作为评估指标，这一指标不太常见，但也是一个有趣的指标。
