from functools import cache
from types import ModuleType

from jinja2 import Environment, PackageLoader, Template


@cache
def _module_env(module: ModuleType) -> Environment:
    loader = PackageLoader(module.__name__)
    return Environment(loader=loader, trim_blocks=True)


def load_template(module: ModuleType, name: str) -> Template:
    """Load template from a module.

    Example:

    ```python
    import chisellm_utils.dataset as dataset_module
    # load a template named `template.j2` from `chisellm_utils.dataset.templates` directory
    template = load_template(dataset_module, "template")
    ```

    Args:
        module (ModuleType): A module contains a subdirectory named `templates`.
        name (str): Template name without extension. e.g. `template.j2` -> `template`

    Returns:
        Template: A jinja2 template object.
    """

    return _module_env(module).get_template(f"{name}.j2")


@cache
def create_template(template: str) -> Template:
    return Environment().from_string(template)
