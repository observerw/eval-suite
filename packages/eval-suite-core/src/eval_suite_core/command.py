import asyncio as aio
import asyncio.subprocess as sp
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from eval_suite_core.exception import BaseEvalResultType, EvalException

type MapExceptionFunc = Callable[[str, str], EvalException]


def default_map_exception(stdout: str, stderr: str) -> EvalException:
    return EvalException(
        message=f"Command failed with error: {stderr or stdout}",
        type=BaseEvalResultType.fail,
    )


@dataclass
class Process:
    """Utility class to run a subprocess and wait for it to finish."""

    process: Awaitable[sp.Process]

    _map_exception: MapExceptionFunc = default_map_exception

    def map_exception(self, map: MapExceptionFunc | EvalException) -> Self:
        match map:
            case EvalException() as exc:
                self._map_exception = lambda _, __: exc
            case Callable() as func:
                self._map_exception = func

        return self

    async def run(self, *, timeout: int = 60) -> str:
        """Run the process and return the result."""
        process = await self.process

        try:
            stdout, stderr = await aio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            if process.returncode != 0:
                raise self._map_exception(stdout.decode(), stderr.decode())

            return stdout.decode().strip()
        except aio.TimeoutError:
            process.kill()
            raise EvalException(
                message=f"Command timed out after {timeout} seconds",
                type=BaseEvalResultType.timeout,
            )


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
            "-v",
            f"{cwd}:{cwd}:rw",
            *work_dir_cmd,
            image,
            *args,
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
            **kwargs,
        )

    @classmethod
    def k8s_run(
        cls,
        *args: str,
        pod: str,
        container: str,
        namespace: str = "default",
        cwd: Path | None = None,
        **kwargs,
    ) -> Process:
        """Run command in a kubernetes pod."""

        if cwd and not cwd.is_absolute():
            raise ValueError("K8s run requires an absolute path for `cwd`.")

        work_dir_cmd = ("-w", str(cwd)) if cwd else ()

        return cls.run(
            "kubectl",
            "exec",
            *work_dir_cmd,
            "-i",
            "-n",
            namespace,
            pod,
            "-c",
            container,
            "--",
            *args,
            **kwargs,
        )

    @classmethod
    def podman_run(
        cls,
        *args: str,
        podman: str = "podman",
        image: str,
        cwd: Path | None = None,
        **kwargs,
    ) -> Process:
        """Run command in a podman container."""

        if cwd and not cwd.is_absolute():
            raise ValueError("Podman run requires an absolute path for `cwd`.")

        work_dir_cmd = ("-w", str(cwd)) if cwd else ()

        return cls.run(
            podman,
            "run",
            "--rm",
            "-i",
            "-v",
            f"{cwd}:{cwd}:rw",
            *work_dir_cmd,
            image,
            *args,
            **kwargs,
        )

    @classmethod
    def podman_exec(
        cls,
        *args: str,
        podman: str = "podman",
        container: str,
        cwd: Path | None = None,
        **kwargs,
    ) -> Process:
        """Run command in an existing podman container."""

        if cwd and not cwd.is_absolute():
            raise ValueError("Podman run requires an absolute path for `cwd`.")

        work_dir_cmd = ("-w", str(cwd)) if cwd else ()

        return cls.run(
            podman,
            "exec",
            *work_dir_cmd,
            "-i",
            container,
            *args,
            **kwargs,
        )
