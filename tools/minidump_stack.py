#!/usr/bin/env python3
"""Print the faulting host stack from a playgta6_boot fault minidump.

Usage: minidump_stack.py <dump.dmp | boot-report.json> [--pdb-dir DIR] [--exe-dir DIR] [--cdb PATH]
                         [--guest-exe PATH] [--no-symbols]

A report argument uses fault_record.minidump. Uses cdb.exe when installed (best: full symbols),
otherwise parses the dump's exception stream and stack memory with the stdlib. Host frames are then
symbolized through Windows' own dbghelp.dll (ctypes) to function+file:line, using the harness
playgta6_boot.exe/.pdb from --exe-dir/--pdb-dir (default: the module path recorded in the dump).
A missing or mismatched PDB degrades to module+offset. --guest-exe names the guest PE: frames inside
its image range are described through tools/guest_symbols.py (names only, never code bytes).
Dumps hold no full memory and stay local: do not commit them.
"""
import argparse, json, ntpath, os, shutil, struct, subprocess, sys

STREAM_THREADS, STREAM_MODULES, STREAM_EXCEPTION, STREAM_MEMORY = 3, 4, 6, 5
STREAM_FAULT_RECORD = 0x50473601  # user stream: guest fault_record JSON (--dump-on-fault)
GUEST_FAULT_CODE = 0xE0475046     # synthetic exception code of guest-fault dumps
HARNESS_IMAGE = "playgta6_boot.exe"


class DbgHelp:
    """Thin ctypes wrapper over dbghelp.dll (Windows only)."""

    def __init__(self, search_path=None):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes
        self.dll = ctypes.WinDLL("dbghelp.dll")
        self.proc = ctypes.c_void_p(-1)  # pseudo handle: no real process behind it
        self.dll.SymSetOptions(0x2 | 0x4 | 0x10)  # UNDNAME | DEFERRED_LOADS | LOAD_LINES
        self.dll.SymInitializeW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.BOOL]
        if not self.dll.SymInitializeW(self.proc, search_path, False):
            raise OSError("SymInitializeW failed")
        self.dll.SymLoadModuleExW.restype = ctypes.c_uint64
        self.dll.SymLoadModuleExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                              ctypes.c_uint64, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        self.dll.SymFromAddrW.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                                          ctypes.c_void_p]
        self.dll.SymGetLineFromAddrW64.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(wintypes.DWORD),
                                                   ctypes.c_void_p]

        class SYMBOL_INFOW(ctypes.Structure):
            _fields_ = [("SizeOfStruct", ctypes.c_uint32), ("TypeIndex", ctypes.c_uint32),
                        ("Reserved", ctypes.c_uint64 * 2), ("Index", ctypes.c_uint32), ("Size", ctypes.c_uint32),
                        ("ModBase", ctypes.c_uint64), ("Flags", ctypes.c_uint32), ("Value", ctypes.c_uint64),
                        ("Address", ctypes.c_uint64), ("Register", ctypes.c_uint32), ("Scope", ctypes.c_uint32),
                        ("Tag", ctypes.c_uint32), ("NameLen", ctypes.c_uint32), ("MaxNameLen", ctypes.c_uint32),
                        ("Name", ctypes.c_wchar * 1024)]

        class IMAGEHLP_LINEW64(ctypes.Structure):
            _fields_ = [("SizeOfStruct", ctypes.c_uint32), ("Key", ctypes.c_void_p), ("LineNumber", ctypes.c_uint32),
                        ("FileName", ctypes.c_wchar_p), ("Address", ctypes.c_uint64)]

        self.SYM, self.LINE = SYMBOL_INFOW, IMAGEHLP_LINEW64

    def load(self, image, base, size):
        return bool(self.dll.SymLoadModuleExW(self.proc, None, image, None, base, size, None, 0))

    def resolve(self, addr):
        """'func+0xoff (file:line)' or None (no symbol)."""
        ct = self.ct
        si = self.SYM()
        si.SizeOfStruct = 88  # sizeof(SYMBOL_INFOW) without the name buffer
        si.MaxNameLen = 1024
        disp = ct.c_uint64(0)
        if not self.dll.SymFromAddrW(self.proc, addr, ct.byref(disp), ct.byref(si)):
            return None
        out = "%s+0x%x" % (si.Name, disp.value)
        ln = self.LINE()
        ln.SizeOfStruct = ct.sizeof(self.LINE)
        d32 = self.wt.DWORD(0)
        if self.dll.SymGetLineFromAddrW64(self.proc, addr, ct.byref(d32), ct.byref(ln)):
            out += " (%s:%d)" % (ln.FileName, ln.LineNumber)
        return out


class Symbolizer:
    """Address -> text. `backend` provides load(image, base, size)->bool and resolve(addr)->str|None
    (DbgHelp, or a mock in tests). `modules` is [(base, size, full_path)]."""

    def __init__(self, modules, backend=None, dirs=(), guest=None, exists=os.path.exists):
        self.modules, self.backend, self.guest = modules, backend, guest
        self.loaded = {}
        self.notes = []
        for base, size, path in modules if backend is not None else ():
            name = ntpath.basename(path)
            if name.lower() != HARNESS_IMAGE:
                continue  # system DLLs would need a symbol server; module+offset is enough for them
            image = next((c for c in [os.path.join(d, name) for d in dirs] + [path] if exists(c)), None)
            if image is None:
                self.notes.append("no %s found (pass --exe-dir): module+offset only" % name)
                continue
            self.loaded[name] = backend.load(image, base, size)
            if not self.loaded[name]:
                self.notes.append("dbghelp could not load %s" % image)

    def describe(self, a):
        for base, size, path in self.modules:
            if base <= a < base + size:
                name = ntpath.basename(path)
                text = "%s+0x%x" % (name, a - base)
                if self.loaded.get(name):
                    r = self.backend.resolve(a)
                    text += "  " + r if r else "  (no symbol: PDB missing or from another build)"
                return text
        if self.guest is not None and self.guest.contains(a):
            return "guest " + self.guest.describe(a)
        return None


def resolve_dump(arg):
    if arg.lower().endswith(".json"):
        with open(arg, encoding="utf-8") as f:
            path = ((json.load(f).get("fault_record") or {}).get("minidump"))
        if not path:
            sys.exit("report has no fault_record.minidump")
        return path
    return arg


def find_cdb(explicit):
    if explicit:
        return explicit
    if os.environ.get("PG6_MINIDUMP_NO_CDB"):  # tests pin the built-in parser's output format
        return None
    found = shutil.which("cdb.exe")
    if found:
        return found
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if root:
            p = os.path.join(root, "Windows Kits", "10", "Debuggers", "x64", "cdb.exe")
            if os.path.exists(p):
                return p
    return None


def parse(data):
    sig, _ver, nstreams, dir_rva = struct.unpack_from("<4sIII", data, 0)
    if sig != b"MDMP":
        sys.exit("not a minidump")
    streams = {}
    for i in range(nstreams):
        t, size, rva = struct.unpack_from("<III", data, dir_rva + 12 * i)
        streams[t] = (size, rva)
    return streams


def read_modules(data, streams):
    modules = []
    if STREAM_MODULES in streams:
        _s, rva = streams[STREAM_MODULES]
        (n,) = struct.unpack_from("<I", data, rva)
        for i in range(n):
            base, size, _cs, _ts, name_rva = struct.unpack_from("<QIIII", data, rva + 4 + 108 * i)
            (nlen,) = struct.unpack_from("<I", data, name_rva)
            modules.append((base, size, data[name_rva + 4:name_rva + 4 + nlen].decode("utf-16-le")))
    return modules


def read_fault_record(data, streams):
    if STREAM_FAULT_RECORD not in streams:
        return None
    size, rva = streams[STREAM_FAULT_RECORD]
    try:
        return json.loads(data[rva:rva + size].decode("utf-8"))
    except ValueError:
        return None


def print_fault_record(fr):
    """Guest-level info from the fault_record user stream."""
    def h(v):
        return "0x%x" % v if isinstance(v, int) else str(v)
    print("guest fault: kind=%s thread=%s rip=%s target=%s function=%s" % (
        fr.get("kind"), fr.get("thread"), fr.get("rip"), fr.get("target"), json.dumps(fr.get("function"))))
    regs = fr.get("registers")
    if isinstance(regs, dict):
        print("  guest registers: " + " ".join("%s=%s" % (k, h(v)) for k, v in regs.items()))
    if fr.get("source") is not None:
        print("  source site: %s" % json.dumps(fr["source"]))
    disp = fr.get("recent_entries")
    if isinstance(disp, list):
        print("  last %d dispatches (newest last):" % len(disp))
        for d in disp[-8:]:
            print("    -> %s %s from %s" % (d.get("target"), d.get("name") or "", d.get("from")))


def make_symbolizer(args, modules):
    guest = None
    if args.guest_exe:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import guest_symbols
        try:
            guest = guest_symbols.PeSymbols.from_path(args.guest_exe)
        except (OSError, ValueError) as e:
            print("guest exe unusable: %s" % e)
    dirs = [d for d in (args.exe_dir, args.pdb_dir) if d]
    backend = None
    if not args.no_symbols and os.name == "nt":
        try:
            backend = DbgHelp(";".join(dirs) or None)
        except OSError as e:
            print("dbghelp unavailable: %s" % e)
    sym = Symbolizer(modules, backend, dirs, guest)
    for n in sym.notes:
        print("note: " + n)
    return sym


def read_threads(data, streams):
    """(thread id, rip, rsp, stack start, stack size, stack rva) per MINIDUMP_THREAD."""
    out = []
    if STREAM_THREADS not in streams:
        return out
    _s, trva = streams[STREAM_THREADS]
    (n,) = struct.unpack_from("<I", data, trva)
    for i in range(n):
        off = trva + 4 + 48 * i
        (t,) = struct.unpack_from("<I", data, off)
        start, size, mrva = struct.unpack_from("<QII", data, off + 24)
        csize, crva = struct.unpack_from("<II", data, off + 40)
        rip = rsp = 0
        if csize >= 0x100:
            rsp = struct.unpack_from("<Q", data, crva + 0x98)[0]
            rip = struct.unpack_from("<Q", data, crva + 0xF8)[0]
        out.append((t, rip, rsp, start, size, mrva))
    return out


def print_all_threads(data, streams, loc, per_thread=16):
    threads = read_threads(data, streams)
    print("%d thread(s)" % len(threads))
    for t, rip, rsp, start, size, mrva in threads:
        print("thread %d rip 0x%x (%s) rsp 0x%x" % (t, rip, loc(rip) or "?", rsp))
        if not (start <= rsp < start + size):
            print("  stack memory not in dump")
            continue
        shown = 0
        for a in range(rsp - start, size - 7, 8):
            (w,) = struct.unpack_from("<Q", data, mrva + a)
            l = loc(w)
            if l:
                print("  [rsp+0x%x] %s" % (a - (rsp - start), l))
                shown += 1
                if shown >= per_thread:
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--pdb-dir")
    ap.add_argument("--cdb")
    ap.add_argument("--exe-dir", help="dir with playgta6_boot.exe + .pdb (default: path recorded in the dump)")
    ap.add_argument("--guest-exe", help="guest PE for naming guest-address frames (guest_symbols.py)")
    ap.add_argument("--no-symbols", action="store_true", help="skip dbghelp symbolization")
    ap.add_argument("--all-threads", action="store_true",
                    help="every thread's candidate return addresses (watchdog / hang dumps have no exception)")
    args = ap.parse_args()
    path = resolve_dump(args.dump)
    cdb = find_cdb(args.cdb)
    with open(path, "rb") as f0:
        head = f0.read()
    fr = read_fault_record(head, parse(head))
    if fr is not None:
        print_fault_record(fr)
    if cdb:
        sym = ".sympath+ " + args.pdb_dir + ";" if args.pdb_dir else ""
        cmd = [cdb, "-z", path, "-c", sym + ("~*kn 40;q" if args.all_threads else ".ecxr;kn 40;q")]
        sys.exit(subprocess.call(cmd))
    data = open(path, "rb").read()
    streams = parse(data)
    modules = read_modules(data, streams)
    if args.all_threads:
        sym = make_symbolizer(args, modules)
        print_all_threads(data, streams, sym.describe)
        return
    if STREAM_EXCEPTION not in streams:
        sys.exit("dump has no exception stream (guest-fault dump): install cdb for thread stacks")
    _s, rva = streams[STREAM_EXCEPTION]
    tid, _pad = struct.unpack_from("<II", data, rva)
    code, _fl, _rec, addr, nparams = struct.unpack_from("<IIQQI", data, rva + 8)
    params = struct.unpack_from("<15Q", data, rva + 8 + 32)
    ctx_size, ctx_rva = struct.unpack_from("<II", data, rva + 8 + 32 + 120)
    rsp = struct.unpack_from("<Q", data, ctx_rva + 0x98)[0]
    rip = struct.unpack_from("<Q", data, ctx_rva + 0xF8)[0]

    sym = make_symbolizer(args, modules)
    loc = sym.describe

    print("exception 0x%08x%s at 0x%x (%s) thread %d" % (
        code, " (guest fault, captured host context)" if code == GUEST_FAULT_CODE else "", addr, loc(addr) or "?", tid))
    if nparams >= 2:
        print("  access violation info: kind=%d address=0x%x" % (params[0], params[1]))
    print("rip 0x%x  rsp 0x%x" % (rip, rsp))
    # Stack memory: the thread list entry holds the stack range captured.
    mem = None
    if STREAM_THREADS in streams:
        _s, trva = streams[STREAM_THREADS]
        (n,) = struct.unpack_from("<I", data, trva)
        for i in range(n):
            off = trva + 4 + 48 * i
            (t,) = struct.unpack_from("<I", data, off)
            if t == tid:
                start, size, mrva = struct.unpack_from("<QII", data, off + 24)
                mem = (start, size, mrva)
    if not mem or not (mem[0] <= rsp < mem[0] + mem[1]):
        print("stack memory not in dump")
        return
    start, size, mrva = mem
    print("candidate return addresses (module+offset  function (file:line)):")
    shown = 0
    for a in range(rsp - start, size - 7, 8):
        (w,) = struct.unpack_from("<Q", data, mrva + a)
        l = loc(w)
        if l:
            print("  [rsp+0x%x] 0x%x %s" % (a - (rsp - start), w, l))
            shown += 1
            if shown >= 40:
                break
    if not any(sym.loaded.values()):
        print("(no symbols loaded: pass --exe-dir <dir with playgta6_boot.exe/.pdb> or use cdb)")


if __name__ == "__main__":
    main()
