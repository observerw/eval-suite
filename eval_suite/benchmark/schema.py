from abc import abstractmethod

from pydantic import BaseModel, PrivateAttr

type InputID = str


class EvalID(BaseModel):
    input_id: InputID
    """Unique identifier of the dataset item"""

    sample_id: int
    """Number of the sample in repeated evaluation"""

    def __str__(self) -> str:
        input_id = self.input_id.replace("/", "_")  # avoids path separator issues
        return f"{input_id}-{self.sample_id}"

    def __hash__(self) -> int:
        return hash(str(self))


class EvalSchema(BaseModel):
    model_config = {"frozen": True}


class EvalInputBase(EvalSchema):
    model_config = {"frozen": True, "extra": "allow"}

    @property
    @abstractmethod
    def input_id(self) -> InputID:
        """Unique identifier of the dataset item"""

    @abstractmethod
    def __str__(self) -> str:
        """Formatted prompt message, will be sent to the model to generate the output"""

    _sample_id: int = PrivateAttr()

    @property
    def _eval_id(self) -> EvalID:
        return EvalID(input_id=self.input_id, sample_id=self._sample_id)


class EvalOutputBase(EvalSchema):
    _eval_id: EvalID = PrivateAttr()
