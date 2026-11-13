import os
import subprocess
from dataclasses import dataclass


class GpuError(Exception):
    pass


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    uuid: str
    memory_mb: int


def parse_nvidia_smi(text):
    gpus = []
    for line in text.strip().splitlines():
        index, name, uuid, mem = [p.strip() for p in line.split(",")]
        gpus.append(Gpu(int(index), name, uuid, int(mem)))
    return gpus


def list_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise GpuError(f"could not list GPUs with nvidia-smi: {e}") from None
    return parse_nvidia_smi(out)


def _listing(gpus):
    return "; ".join(f"{g.index}: {g.name} ({g.memory_mb} MB, {g.uuid})" for g in gpus)


def resolve(ref, gpus):
    if isinstance(ref, str) and ref.strip().isdigit() and any(g.index == int(ref) for g in gpus):
        ref = int(ref)
    if isinstance(ref, int):
        found = [g for g in gpus if g.index == ref]
        if not found:
            raise GpuError(f"no GPU number {ref}. GPUs found: {_listing(gpus)}")
        return found[0]
    if ref.upper().startswith("GPU-"):
        found = [g for g in gpus if g.uuid.lower() == ref.lower()]
        if not found:
            raise GpuError(f"no GPU with UUID {ref}. GPUs found: {_listing(gpus)}")
        return found[0]
    found = [g for g in gpus if ref.lower() in g.name.lower()]
    if not found:
        raise GpuError(f"no GPU matches '{ref}'. GPUs found: {_listing(gpus)}")
    if len(found) > 1:
        raise GpuError(f"'{ref}' matches several GPUs; use a UUID. GPUs found: {_listing(gpus)}")
    return found[0]


def select(specs, gpu_ref, gpus):
    placed = [(s, resolve(s.gpu, gpus)) for s in specs]
    if gpu_ref is not None:
        target = resolve(gpu_ref, gpus)
        mine = [s for s, g in placed if g == target]
        if not mine:
            raise GpuError(f"no models in the config are on {target.name} ({gpu_ref!r})")
        return target, mine
    targets = {g for _, g in placed}
    if len(targets) > 1:
        names = ", ".join(sorted(g.name for g in targets))
        raise GpuError(f"the config places models on several GPUs ({names}); start one server per GPU "
                       f"with --gpu (the stage 2 router will do this for you)")
    return placed[0][1], specs


def pin(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu.uuid
