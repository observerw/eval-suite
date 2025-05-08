# Metrics

<!-- 
eval-suite提供了一组预先定义的 metrics 可供使用。在 eval-suite 中，metrics并不是一个预先封装好的、不可更改的函数，而是一组互相关联的组件和方法。根据metric的特性，其提供的内容也会有所不同。例如：[Pass@k Metric](eval_suite/metrics/pass_k.py) 提供了：

- `EvalConfig`，其中新增了 `k` 配置项；
- `EvalResult`，要求评估结果中必须包含 `passed` 字段，用来表示该样本是否通过单元测试；
- `Stat`，用于在 `EvalResult` 的基础上对结果进行统计，给出从 Pass@1 到 Pass@k 的统计结果；

可以看到，这些组件是相互依赖的关系，为了能够给出最终的 `Stat`，Pass@k Metric 对于 `EvalResult` 进行了约束，从而能够对结果进行统计；同时对 `EvalConfig` 进行了扩展，提供了必要的额外配置项。

更加复杂的 Metric 可能涉及到对方法的默认实现，如 [LLM-as-a-judge](eval_suite/metrics/llm_judge.py) 提供了：

- `EvalOutput` 要求用户提供的评估输出中必须包含 `content` 字段，即需要LLM给出评分的内容；
- `EvalResult` 中包含了 `score` 字段，表示LLM给出的评分；
- `Benchmark.to_result_batch_async` 方法给出了一个实现，调用 `judge` LLM 对结果进行评分，因此用户无须手动实现调用 Judge LLM 的逻辑；
- 新增了对 `to_judge_result` 方法的要求，用户必须手动实现该方法，从 Judge LLM 的输出中提取出评分结果；

通过如上两个例子，我们可以总结出 eval-suite 中预定义 metric 的特性：

- Metric 可能会提供 `EvalInput`（不太常见），`EvalOutput`，`EvalResult`，`EvalStat`，`EvalConfig` 等 schema，这些schema 中包含了若干必须的字段；
- Metric 可能会提供默认实现的方法，如 `to_result_batch_async`，用户可以使用这些方法来得到相应的结果；
 -->

## Composing Multiple Metrics
