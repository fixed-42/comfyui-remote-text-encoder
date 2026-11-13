import time
from collections import OrderedDict, deque

from server_status import log_line

THRASH_SWAPS = 4
THRASH_WINDOW_S = 60


class DoesNotFit(Exception):
    pass


class ModelCache:
    def __init__(self, load, unload, free_mb, file_mb, reserve_mb=2048, oom=(MemoryError,), gpu="the GPU",
                 log=print, clock=time.monotonic):
        self._load, self._unload, self._free_mb, self._file_mb = load, unload, free_mb, file_mb
        self.reserve_mb, self.oom, self.gpu, self.log, self.clock = reserve_mb, oom, gpu, log, clock
        self._entries = OrderedDict()
        self.kept = set()
        self._used = {}
        self._recent = deque()
        self._warned = None
        self.swaps = 0
        self.last_swap = None

    def loaded(self):
        now = self.clock()
        entries, used = list(self._entries.items()), dict(self._used)
        return [(f, t, e, now - used.get((f, t), now)) for (f, t), e in entries]

    def get(self, file, typ):
        key = (file, typ)
        if key in self._entries:
            self._entries.move_to_end(key)
            self._used[key] = self.clock()
            return self._entries[key]
        need = self._file_mb(file)
        at_best = self._free_mb() + sum(getattr(e, "vram_mb", 0) for k, e in self._entries.items()
                                        if k not in self.kept)
        if at_best < need + self.reserve_mb:
            raise self._does_not_fit(file, need, at_best)
        evicted = []
        while self._free_mb() < need + self.reserve_mb and self._evictable():
            evicted.append(self._evict_oldest())
        free = self._free_mb()
        if free < need + self.reserve_mb:
            raise self._does_not_fit(file, need, free)
        t0 = self.clock()
        entry = self._load_once_more_on_oom(file, typ, evicted)
        ms = round((self.clock() - t0) * 1000)
        self._entries[key] = entry
        self._used[key] = self.clock()
        self._record(f"{file} as {typ}", evicted, ms, getattr(entry, "vram_mb", 0))
        return entry

    def _load_once_more_on_oom(self, file, typ, evicted):
        try:
            return self._load(file, typ)
        except self.oom as e:
            if not self._evictable():
                raise DoesNotFit(f"'{file}' as {typ} ran out of VRAM while loading on {self.gpu}: {e}") from None
        while self._evictable():
            evicted.append(self._evict_oldest())
        try:
            return self._load(file, typ)
        except self.oom as e:
            raise DoesNotFit(f"'{file}' as {typ} ran out of VRAM while loading on {self.gpu}: {e}") from None

    def _does_not_fit(self, file, need, free):
        kept_mb = sum(getattr(e, "vram_mb", 0) for k, e in self._entries.items() if k in self.kept)
        if kept_mb:
            return DoesNotFit(f"'{file}' needs {need} MB + {self.reserve_mb} MB working memory; {free} MB free on "
                              f"{self.gpu} beside the kept models ({kept_mb} MB); uncheck one in Remote Encoder "
                              f"Status")
        return DoesNotFit(f"'{file}' needs {need} MB + {self.reserve_mb} MB working memory; {free} MB free on "
                          f"{self.gpu} even with nothing loaded")

    def _evictable(self):
        return [k for k in self._entries if k not in self.kept]

    def _evict_oldest(self):
        key = self._evictable()[0]
        entry = self._entries.pop(key)
        del self._used[key]
        self._unload(entry)
        return f"{key[0]} as {key[1]}"

    def unload(self, file, typ):
        key = (file, typ)
        if key not in self._entries:
            return False
        entry = self._entries.pop(key)
        del self._used[key]
        self._unload(entry)
        self.log(f"unloaded {file} as {typ} (no longer kept)")
        return True

    def unload_idle(self, max_idle_s, why=""):
        now = self.clock()
        gone = []
        for key in [k for k in self._entries if k not in self.kept and now - self._used.get(k, now) >= max_idle_s]:
            entry = self._entries.pop(key)
            del self._used[key]
            self._unload(entry)
            gone.append(f"{key[0]} as {key[1]}")
            self.log(f"unloaded {key[0]} as {key[1]} (idle {max_idle_s} s{why})")
        return gone

    def _record(self, label, evicted, ms, vram_mb):
        self.swaps += 1
        self.last_swap = {"loaded": label, "evicted": evicted, "ms": ms,
                          "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.log(log_line("SWAP", loaded=label, evicted=",".join(evicted), ms=ms, vram=vram_mb,
                          free=self._free_mb()))
        now = self.clock()
        self._recent.append(now)
        while self._recent and now - self._recent[0] >= THRASH_WINDOW_S:
            self._recent.popleft()
        if len(self._recent) > THRASH_SWAPS and (self._warned is None or now - self._warned >= THRASH_WINDOW_S):
            self._warned = now
            self.log(f"WARNING: frequent model switching ({len(self._recent)} swaps in {THRASH_WINDOW_S} s); "
                     f"the models in use don't fit together on {self.gpu}")
