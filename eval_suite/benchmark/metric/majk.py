from pydantic import RootModel


class MajKStat(RootModel):
    root: dict[str, float]


class MajKBenchmark:
    pass
