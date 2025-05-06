import asyncio as aio
import asyncio.subprocess as sp
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path

from eval_suite.exception import BaseEvalResultType, EvalException

logger = logging.getLogger(__name__)


@dataclass
class Process:
    """Utility class to run a subprocess and wait for it to finish."""

    process: Awaitable[sp.Process]

    async def run(
        self,
        *,
        timeout: int = 60,
        exc: EvalException = EvalException(),
    ) -> str:
        """Run the process and return the result."""
        process = await self.process

        try:
            stdout, stderr = await aio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            if process.returncode != 0:
                if not exc.message:
                    exc.message = (
                        f"Command failed with error: {(stderr or stdout).decode()}"
                    )
                raise exc
            return stdout.decode().strip()
        except aio.TimeoutError:
            process.kill()
            if not exc.message:
                exc.message = f"Command timed out after {timeout} seconds"
            if not exc.type:
                exc.type = BaseEvalResultType.timeout
            raise exc
        except BaseException as e:
            process.kill()
            if not exc.type:
                exc.type = BaseEvalResultType.fail
            exc.exc = e
            raise exc


class CommandBase:
    """Specify how to run command line tools."""

    @classmethod
    def run(
        cls,
        *cmd: str,
        cwd: Path | None = None,
        **kwargs,
    ) -> Process:
        return Process(
            sp.create_subprocess_exec(
                *cmd,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                cwd=cwd,
                **kwargs,
            )
        )

    @classmethod
    def docker_run(
        cls,
        *args: str,
        image: str,
        cwd: Path | None = None,
        **kwargs,
    ) -> Process:
        """Run command in a docker container."""

        if cwd and not cwd.is_absolute():
            raise ValueError("Docker run requires an absolute path for `cwd`.")

        work_dir_cmd = ("-w", str(cwd)) if cwd else ()

        return cls.run(
            "docker",
            "run",
            "--rm",
            "-i",
            image,
            "-v",
            f"{cwd}:{cwd}:rw",
            *work_dir_cmd,
            *args,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            **kwargs,
        )

    @classmethod
    def docker_exec(
        cls,
        *args: str,
        container: str,
        cwd: Path | None = None,
        **kwargs,
    ) -> Process:
        """Run command in an existing docker container."""

        if cwd and not cwd.is_absolute():
            raise ValueError("Docker run requires an absolute path for `cwd`.")

        work_dir_cmd = ("-w", str(cwd)) if cwd else ()

        return cls.run(
            "docker",
            "exec",
            *work_dir_cmd,
            "-i",
            container,
            *args,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            **kwargs,
        )
