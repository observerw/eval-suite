import ray


@ray.remote
def a():
    return 1


@ray.remote
def b():
    return 2


@ray.remote
def c(a: int, b: int):
    return a + b


c_ref = c.bind(a.bind(), b.bind())
print(ray.get(c_ref.execute()))
