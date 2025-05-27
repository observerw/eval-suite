from typing import NamedTuple, NewType

ItemID = NewType("ItemID", str)
SampleID = NewType("SampleID", int)
MetricID = NewType("MetricID", str)


class EvalID(NamedTuple):
    item: ItemID
    sample: SampleID


class MetricEvalID(NamedTuple):
    item: ItemID
    sample: SampleID
    metric: MetricID
