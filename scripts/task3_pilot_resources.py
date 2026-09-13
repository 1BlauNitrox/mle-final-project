"""Resource/hash helpers reused from issue #168 at dcff79f."""

import hashlib
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def network_sha(network):
    result = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        result.update(name.encode() + tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def sample_tree(process, cpu_by_identity):
    import psutil

    memory = 0
    for p in [process, *process.children(recursive=True)]:
        try:
            t = p.cpu_times()
            key = (p.pid, p.create_time())
            cpu_by_identity[key] = max(cpu_by_identity.get(key, 0), t.user + t.system)
            memory += p.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return sum(cpu_by_identity.values()), memory


def stop_owned(child):
    import psutil

    try:
        parent = psutil.Process(child.pid)
        for p in parent.children(recursive=True):
            p.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    child.wait()
