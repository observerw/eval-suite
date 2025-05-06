# LLM Evaluation Suite

A flexible, composable, and extensible LLM evaluation framework

> Possibly the most modern LLM evaluation framework you've ever seen 😋

## Motivation & Advantages

Existing LLM evaluation frameworks come with many limitations.

We've observed several key issues:

- Many "implicit" components that are difficult to use or modify;
- Lack of reasonable extensible design, often requiring waiting for upstream support to implement new evaluation tasks/datasets;
- Excessive pursuit of "configuration-only usage", forcing logic to be implemented through configuration;
- Difficulty integrating with existing LLM ecosystems.

To address these issues, we redesigned an LLM evaluation framework aimed at providing better flexibility and extensibility. Core advantages include:

- Complete flexibility: Support for custom evaluation tasks, models, metrics, statistical results, etc.;
- Extensibility: Through reasonable design abstractions, you can easily support new evaluation tasks/datasets without waiting for upstream support;
- Multi-model support: Compatible with `OpenAI`-compatible servers, `SGLang` offline/online/basic interfaces, `vLLM` offline/online interfaces, etc.;
- Comprehensive evaluation experience optimization: Support for evaluation result caching, automatic request retry, and other features to maximize evaluation efficiency and stability.

**The framework design is based on the principle of composition over inheritance**: The evaluation framework's components are decoupled, not forcing you to use awkward configuration parameters to implement features that should be easy to implement, such as dataset filtering or grading before evaluation. Implement the right functionality in the right way, then combine them together to form a complete evaluation process.

## Design Philosophy

We abstract the model evaluation process as: **execute multiple generations and evaluations for each sample in the dataset; aggregate the evaluation results to obtain the final overall evaluation result**.

> So, if you don't agree with the evaluation approach above, or if you want to implement a more complex evaluation process, this framework may not be the best choice for you.

Therefore, we have defined the following interfaces according to the evaluation stages:

Generation and processing flow:

- `to_input`: Convert data items from the dataset to evaluation inputs.
- `to_output`: Convert model-generated results to evaluation outputs.
- `to_result`: Convert evaluation outputs to evaluation results.
- `to_stat`: Convert evaluation results to statistical information.

Each interface can access all context information from previous stages.

## Example: Evaluating Pass@k Metrics on the HumanEval Dataset

See the [complete example](examples/humaneval.py) for details.

### Step 1: Construct the evaluation input `EvalInput`

Two methods must be implemented:

- `input_id`: A unique identifier for the evaluation input, usually the id from the dataset.
- `__str__`: String representation of the evaluation input, specifying how to convert it to prompt text.

```python
class EvalInput(EvalInputBase):
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    @property
    def input_id(self) -> str:
        return self.task_id

    def __str__(self) -> str:
        return self.prompt
```

Then implement the `to_input` method to convert raw data to our defined `EvalInput`. Since we use `pydantic` to define data structures, we can directly use the `model_validate` method to convert raw data to an `EvalInput` object:

```python
class HumanEvalBenchmark(passk.Benchmark[EvalInput]):
    @classmethod
    def to_input(cls, ctx: passk.InputContext) -> EvalInput:
         return EvalInput.model_validate(ctx.data)
```

If this method is not implemented, the default behavior is to use `model_validate` for conversion.

### Step 2: Construct `EvalOutput` from model generation results

```python
class HumanEvalBenchmark(passk.PassKBenchmark[EvalInput]):
    @staticmethod
    async def to_output(ctx: passk.OutputContext[EvalInput]) -> passk.EvalOutput:
        return passk.EvalOutput(code=ctx.generation.content)
```

The `to_output` method is typically used to convert the model's generation results to another format, such as extracting answers from the output (hint: you can use the [utility classes](#utility-classes) to easily implement these functions).

### Step 3: Evaluate model outputs to get `EvalResult`

```python
class HumanEvalBenchmark(passk.PassKBenchmark[EvalInput]):
    @staticmethod
    async def to_result_async(ctx: passk.ResultContext[EvalInput]) -> passk.EvalResult:
        _ = await cmd.python(code=ctx.output.code, cwd=ctx.eval_path).run()
        return passk.EvalResult(passed=True)
```

Alternatively, if you think batch evaluation can improve efficiency (for example, if you need to call a Judge LLM for batch scoring during evaluation), you can also implement the `to_result_batch_async` method. See [Choosing the appropriate evaluation process implementation based on characteristics](#choosing-the-appropriate-evaluation-process-implementation-based-on-characteristics).

### Step 4: Aggregate evaluation results to get `EvalStat`

```python
class PassKBenchmark(...):
    @staticmethod
    def stat(ctx: StatContext) -> EvalStat:
        return EvalStat.init(groups=ctx.groups, k=ctx.config.k)
```

For each evaluation method, we expect to see certain statistical data, such as `pass@1`, `pass@10`, etc., in Pass@k evaluations. These default statistics will be implemented in the form of `EvalStat`. If you don't need additional statistics, you can directly use the default implementation of `EvalStat`.

### Getting Results

Looks pretty good, right?

## Statistical Results

### Generic Statistics

### Let LLM Help You with Statistics

## Choosing the Appropriate Evaluation Process Implementation Based on Characteristics

- `to_result`: Computationally intensive tasks that can be executed independently, such as word frequency statistics on generated text. Ignores `eval_batch_size` and uses processes equal to the number of CPUs to execute evaluation tasks.
- `to_result_async`: IO-intensive tasks that can be executed independently, such as calling external APIs for evaluation results. Controls the number of concurrent evaluation tasks through `eval_batch_size`.
- `to_result_batch`: Batch-executed computationally intensive tasks, such as batch tokenization of generated text. Not commonly used; in most cases, implementing `to_result` is sufficient.
- `to_result_batch_async`: Batch-executed IO-intensive tasks, such as using LLM to batch-score generated content. Typically used when dealing with external APIs that have built-in batch processing.

## Loading Benchmark Temporary Resources, Such as Output Paths

<!-- Benchmarks can be used within a 'with' context, allowing for temporary resource loading -->

<!-- You can manually specify `base_path`; if not specified, you need to use a context manager to automatically load a temporary directory. -->

## Benchmark Configuration

Benchmark configuration can be set through the `BaseEvalConfig` class. Here are all available configuration options:

| Configuration   | Type          | Default Value   | Description                                                                                                           |
| --------------- | ------------- | --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `overwrite`     | `bool`        | `False`         | Whether to overwrite existing evaluation results                                                                      |
| `use_cache`     | `bool`        | `True`          | Whether to use cached evaluation data                                                                                 |
| `n_samples`     | `int`         | `1`             | Number of generations per sample                                                                                      |
| `max_n_samples` | `int \| None` | `n_samples * 2` | Maximum number of generations per sample until at least `n_samples` valid results are obtained. `None` means no limit |
| `system_prompt` | `str \| None` | `None`          | System prompt used during evaluation                                                                                  |
| `overlap`       | `bool`        | `True`          | Whether to overlap sample generation and sample evaluation processes                                                  |

## Caching Evaluation Results

Calling commercial model APIs can be expensive, so you typically want to preserve completed generation/evaluation results even if the evaluation process encounters errors. To address this, we provide a simple caching mechanism.

## Exception Handling

During evaluation, `raise EvalException` will be caught and recorded in the final results.

In most cases, exceptions are not what we want; we want to obtain valid evaluation results for statistics. Therefore, the framework will re-evaluate processes that throw exceptions until enough valid evaluation results are obtained (or until the `max_n_samples` limit is reached, see [Benchmark Configuration](#benchmark-configuration)). If you want to throw an exception during evaluation, you can use `raise EvalException`.

However, in some evaluations, exceptions are also an expected evaluation result. For example, in Pass@k evaluation, a failed process execution indicates that the generated result did not pass the test, which is expected. For such cases, we can implement the `EvalResultBase.from_exception` method to create evaluation results from exceptions:

```python
class EvalResult(EvalResultBase):
    @classmethod
    def from_exception(cls, exc: EvalException) -> EvalResult:
        # Continue to throw unknown exceptions to avoid masking unexpected behavior
        if exc.type == BaseEvalResultType.fail:
            raise exc

        # Expected exception, return evaluation result
        return cls(
            passed=False,
            exception=exc,
        )
```

## Building New Benchmarks

## Performance Optimization Tips

- Set evaluation parameters reasonably based on your hardware resources.
- `to_input`, `to_output` will be called multiple times, so they should be lightweight, idempotent functions.

## Executing External Commands

A common requirement for LLM evaluation is to execute external commands to obtain evaluation results, such as calling a compiler to compile and run code generated by the model. For this, we provide a convenient solution.

## Model Clients

### Client Configuration

### (OpenAI) Managing Docker Container Lifecycle

When initializing an `OpenAIClient`, you can optionally specify a `docker_container_id` parameter. This allows you to manage the lifecycle of the Docker container used for evaluation. The container will start when the Client is initialized and stop when the Client is closed.

```python
from lm_eval.client import OpenAIClient

with OpenAIClient(
    # ...
    docker_container_id="my_container",
) as client:
    # ✅ Docker container will start when Client is initialized
    _ = client.generate( ... )
# ✅ Docker container will automatically stop when Client is closed
```

Tips: We know that after starting a Docker container, it may take some time for the container to fully start. Therefore, during Client initialization, a function called `wait_for_openai_server` is called, repeatedly requesting the `/models` interface until the model list includes the model specified by the Client.

### (SGLang) Delayed Initialization

SGLang `Engine` automatically initializes the model when created, which may take some time. Therefore, we delay initialization until the first call to `generate` rather than initializing when entering the `with` block.

```python
from lm_eval.client import SGLangClient

model = SGLangClient( ... )

with model as client:
    # ✅ At this point, `generate` has not been called, so the model is not initialized
    if not do_some_check():
        return

    # ✅ Now `generate` is called, so the model will be initialized
    _ = client.generate( ... )
```

### (SGLang) `<think>` Tag Completion

Many reasoning models use the trick of adding `<think>` tags to generation prompts to force the model to think. However, this can disrupt reasoning parsing. Therefore, we apply a simple trick: when there is a `</think>` tag in the response, we try to add a `<think>` tag at the beginning.

## Utility Classes

### Extracting Generation Results

````python
result = extract_code(code)

code = """
```python
# code block a
```

```python {single}
# code block b
```

```python [grouped] {a}
# code block c
```

```python [grouped] {b}
# code block d
```

```scala
// code block e
```
"""

all_codes: list[str] = result.get('python') # for all blocks marked with python
assert all_codes == ["# code block a", "# code block b", "# code block c", "# code block d"]

single_code: str = result.get('python', id='single') # python and <single>
assert single_code == "# code block b"

grouped_codes: list[str] = result.get('python', group='grouped') # python and [grouped]
assert grouped_codes == ["# code block c", "# code block d"]

code_inside_group: str = result.get('python', group='grouped', id='b') # python and [grouped] and <c>
assert code_inside_group == "# code block d"
````

### Loading Jinja2 Templates

Jinja2 templates are a great way to maintain prompts. We provide a simple utility class to load Jinja2 templates, which you can then render as strings to use as model prompts:

```python
from lm_eval.utils.template import load_template
import curr_module

# curr_module.template.some_template.j2 will be loaded
template = load_template(curr_module, "some_template")
prompt = template.render(
    # Pass template variables
    var1="value1",
    var2="value2",
)
```

### Loading Huggingface Datasets

Yes, we all love `datasets`, but its support for type hinting is really poor:

```python
from datasets import load_dataset

dataset = load_dataset("human_eval", split="test")
# dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset 😅
```

To enhance type hinting support, when you clearly know that the data type you're loading is indeed a `datasets.Dataset`, you can use the `lm_eval.utils.dataset.load_dataset` function to load the dataset. This function automatically infers the type of the dataset and returns a `datasets.Dataset` object.

```python
from lm_eval.utils.dataset import load_dataset

dataset = load_dataset("human_eval", split="test") # load from repo_id
dataset = load_dataset(Path("path/to/dataset")) # load from local disk
# dataset: Dataset
```
