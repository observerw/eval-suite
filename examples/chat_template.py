from pathlib import Path

from eval_suite_core import prompt
from eval_suite_core.prompt import pt
from eval_suite_core.prompt.utils import create_template
from eval_suite_kit.prompt.few_shot import FewShotFormatter

few_shot_formatter = FewShotFormatter(
    name="few_shot",
    n_shots=3,
    examples=[
        "This is an example of a few-shot formatter.",
        "Here is another example to illustrate the concept.",
        "This example shows how to use few-shot learning in prompts.",
        "A different example that demonstrates the application of few-shot techniques.",
        "This example provides additional context for few-shot learning.",
    ],
)

few_shot_template = create_template("""
Here are some examples to guide you: 

{% for example in examples %}
Example {{ loop.index }}: {{ example }}
{% endfor %}
""")

chat_template = prompt.ChatTemplate.compose(
    pt.system("You are a helpful assistant."),
    pt.user("Here's your task: {{task}}"),
    pt.user(Path("prompts/extra-context.j2")),
    pt.user("The following is the chat history: "),
    pt.history_placeholder,
    pt.user(few_shot_template),
    pt.placeholder("task"),
    pt.user(
        [
            "Please provide a detailed description of the following images:",
            pt.Image("https://example.com/image1.png"),
            pt.Image(Path("images/image2.jpg")),
            pt.Image(bytes(Path("images/image3.png").read_bytes())),
        ]
    ),
)

partial_template = (
    chat_template
    | few_shot_formatter
    | {"task": "Analyze the images and provide insights based on the examples given."}
)
