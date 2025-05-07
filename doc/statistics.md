# Statistics

<!-- 与设计理念保持一致，eval-suite给出的评估结果也是可组合的。 -->

## Metric-Specific Statistics

<!-- 
 Metric 是一个综合性的概念，可能包含多方面的要求。例如，有的 metric 基本上只要求了对于评估结果的计算方法，但并不强制要求该如何对结果进行统计（如Bleu Score等）；而有些 metric 则要求在在特定的评估结果上进行特殊的统计，最典型的例子即为 Pass@k。

 因此，对统计方式具有要求的 metric 通常会提供若干统计类，这些统计类必须只能在该 metric 特定的评估结果上使用。 例如，Pass@k metric 提供了 `PassKStat` 统计类，该类只能在 `PassKResult` 评估结果上使用。 这些统计类通常会在 metric 的文档中进行说明。
 -->

## Related Files

<!-- 
某些统计类可能会提供额外的文件作为统计结果的补充，如图表、数据表等。 这些统计类中将包含一个名为 `files` 的字段，其中包含了所有额外文件的名称、描述和路径。

通常而言，这些统计类的 `from_groups` 方法中将会包含一个可选的 `files_path` 参数， 该参数用于指定额外文件的存放路径。 如果你希望保存额外文件，则可以在 `from_groups` 方法中传入该参数。 
-->

## Exclude Fields in Statistics

<!-- 
出于某种原因，你可能并不需要预先给定的统计类中的某些字段（这些字段太长或不相关），这时你可以将它们排除掉： 
-->

```python
from pydantic import Field
from eval_suite.benchmark.stat.score import ScoreStat

# Step ️1: we derive a new class from the original stat class
class ExcludedScoreStat(ScoreStat):
    # Step 2: we explicitly set the `exclude` parameter to True
    scores: list[float] = Field(exclude=True)
```

<!-- 值得注意的是，排除字段将会导致输出结果无法反序列化（除非你将字段设置为 optional）；此时重复运行同一个 benchmark 的 `run` 方法将不会返回统计类。 -->

## Create your own Statistics Class

<!-- 
 统计类通常是灵活多变的，因此 eval-suite 并没有过于限制该如何创建一个统计类；唯一的限制为必须继承 `EvalStatBase` 类。
 
 我们推荐遵循如下的最佳实践：

 1. 具有一个名为 `from_groups` 的类方法，至少接受一个 `EvalStatGroups` 类型的参数；适当的添加额外的必需参数。
 2. 如果需要输出额外文件，则在 `from_groups` 方法中添加一个可选的 `files_path` 参数。
 3. 如果统计必须在某类特定的 `EvalResult` 上进行，则在定义统计类时可以添加一个 `[Result: EvalResultBase]` 的类型参数，并在 `from_groups` 方法中定义 `groups: EvalStatGroups[Result]` 参数。
 
 实现统计类时需要注意以下几点：

 1. 统计类必须是可序列化的，因此不应包含任何不可序列化的字段（如文件句柄、数据库连接等）。
 2. 统计类应提供清晰的文档，以便用户理解其功能和用法。
 
 -->