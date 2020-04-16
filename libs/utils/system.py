# TODO: add cross-platform method. This is UNIX-only
MEMINFO = "/proc/meminfo"
GIG = 1024**2

def get_memory_usage():
    try:
        with open(MEMINFO) as file:
            a = {l.split()[0][:-1]: int(l.split()[1]) for l in file}
            used = a["MemTotal"] - a["MemFree"] - a["Buffers"] - a["Cached"] - a["Slab"]
        return used
    except FileNotFoundError:
        raise NotImplementedError("Function ")

def check_memory_limit(max_size, verbose=True):
    used = get_memory_usage()
    ok = used < max_size
    if verbose:
        status = "OK" if ok else "NOT OK"
        print(f"Memory usage {status}: {used/GIG:.1f}G / {max_size/GIG:.1f}G")
    return ok