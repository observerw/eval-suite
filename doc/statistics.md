# Statistics

<!-- 与设计理念保持一致，eval-suite给出的评估结果也是可组合的。 -->

## Exclude Fields in Statistics

<!-- 出于某种原因，你可能并不需要预先给定的统计类中的某些字段（这些字段太长或不相关），这时你可以将它们排除掉： -->

```python
from pydantic import Field
from eval_suite.benchmark.stat.score import ScoreStat

class ExcludedScoreStat(ScoreStat):
    # exclude the field `scores` using pydantic `Field` 
    scores: list[float] = Field(exclude=True)
```

<!-- 值得注意的是，排除字段将会导致输出结果无法反序列化（除非你将字段设置为 optional）；此时重复运行同一个 benchmark 的 `run` 方法将不会返回统计类。 -->