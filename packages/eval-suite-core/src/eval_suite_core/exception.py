from enum import Enum


class BaseEvalResultType(str, Enum):
    success = "success"
    """Evaluation completed successfully"""

    fail = "fail"
    """General/unknown failure"""

    timeout = "timeout"
    """Timeout"""


class EvalException(Exception):
    """Expected error during evaluation, will be a part of the result"""

    message: str | None = None
    type: str
    exc: BaseException | None = None

    def __init__(
        self,
        *,
        message: str | None = None,
        type: str = BaseEvalResultType.fail,
        exc: BaseException | None = None,
    ):
        super().__init__(message)

        self.type = type
        self.message = message
        self.exc = exc

    @classmethod
    def from_exception(cls, exc: BaseException) -> "EvalException":
        match exc:
            case EvalException():
                return exc
            case other:
                return cls(exc=other, message=str(other))


class EvalError(Exception):
    """Unxpected error during evaluation"""
