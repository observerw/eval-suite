from eval_suite_core.client import (
    OfflineClientBase,
    OnlineClientBase,
    SamplingParamsBase,
)


class SGLangNativeSamplingParams(SamplingParamsBase): ...


class SGLangNativeClient(OnlineClientBase): ...


class SGLangEngineClient(OfflineClientBase): ...
