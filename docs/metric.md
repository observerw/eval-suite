# Metric

## DAG

## Load Resources

<!--
在某些情况下，你定义的 Metric 需要一些临时资源才能够正常进行评测，在 Python 中这通常使用 `with` 上下文管理器实现。

为了保持接口的一致性，我们为
-->

For loading contextual resources, it is suggested to override the `init` method of the `Metric` class:
