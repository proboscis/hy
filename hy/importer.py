import builtins
import importlib
import inspect
import json
import os
import pkgutil
import sys
import types
import zipimport
from contextlib import contextmanager
from functools import partial

import hy
from hy.compiler import hy_compile
from hy.reader import read_many, HyReader


@contextmanager
def loader_module_obj(loader):
    """Use the module object associated with a loader.

    This is intended to be used by a loader object itself, and primarily as a
    work-around for attempts to get module and/or file code from a loader
    without actually creating a module object.  Since Hy currently needs the
    module object for macro importing, expansion, and whatnot, using this will
    reconcile Hy with such attempts.

    For example, if we're first compiling a Hy script starting from
    `runpy.run_path`, the Hy compiler will need a valid module object in which
    to run, but, given the way `runpy.run_path` works, there might not be one
    yet (e.g. `__main__` for a .hy file).  We compensate by properly loading
    the module here.

    The function `inspect.getmodule` has a hidden-ish feature that returns
    modules using their associated filenames (via `inspect.modulesbyfile`),
    and, since the Loaders (and their delegate Loaders) carry a filename/path
    associated with the parent package, we use it as a more robust attempt to
    obtain an existing module object.

    When no module object is found, a temporary, minimally sufficient module
    object is created for the duration of the `with` body.
    """
    tmp_mod = False

    try:
        module = inspect.getmodule(None, _filename=loader.path)
    except KeyError:
        module = None

    if module is None:
        tmp_mod = True
        module = sys.modules.setdefault(loader.name, types.ModuleType(loader.name))
        module.__file__ = loader.path
        module.__name__ = loader.name

    try:
        yield module
    finally:
        if tmp_mod:
            del sys.modules[loader.name]


def _hy_code_from_file(filename, loader_type=None):
    """Use PEP-302 loader to produce code for a given Hy source file."""
    full_fname = os.path.abspath(filename)
    fname_path, fname_file = os.path.split(full_fname)
    modname = os.path.splitext(fname_file)[0]
    sys.path.insert(0, fname_path)
    try:
        if loader_type is None:
            loader = pkgutil.get_loader(modname)
        else:
            loader = loader_type(modname, full_fname)
        code = loader.get_code(modname)
    finally:
        sys.path.pop(0)

    return code


def _get_code_from_file(*args, hy_src_check=lambda x: x.endswith(".hy")):
    """A patch of `runpy._get_code_from_file` that will also run and cache Hy
    code.
    """

    if hy.compat.PY3_15:
        fname, module = args
    elif hy.compat.PY3_12:
        fname, = args
    else:
        run_name, fname = args
        if fname is None and run_name is not None:
            fname = run_name

    # Check for bytecode first.  (This is what the `runpy` version does!)
    with open(fname, "rb") as f:
        code = pkgutil.read_code(f)

    if code is None:
        if hy_src_check(fname):
            code = _hy_code_from_file(fname, loader_type=HyLoader)
        else:
            # Try normal source
            with open(fname, "rb") as f:
                # This code differs from `runpy`'s only in that we
                # force decoding into UTF-8.
                source = f.read().decode("utf-8")
            code = compile(
                source, fname, "exec",
                **(dict(module=module) if hy.compat.PY3_15 else {}))

    return code if hy.compat.PY3_12_6 else (code, fname)


importlib.machinery.SOURCE_SUFFIXES.insert(0, ".hy")
_py_source_to_code = importlib.machinery.SourceFileLoader.source_to_code


def _could_be_hy_src(filename):
    return (
        os.path.splitext(filename)[1]
        not in set(importlib.machinery.SOURCE_SUFFIXES) - {".hy"}
    )


# Bytecode compiled from Hy source depends not only on that source, but
# also on the macros it was expanded with. Python's own staleness check
# for a bytecode file looks only at the modification time and size of
# the source, so after a change to a macro, modules that use the macro
# would keep running their old expansions. To prevent this, we append to
# each bytecode file a record of the files that the module's macros came
# from, and recompile when the record is missing or no longer matches.
# (`marshal.loads` ignores trailing bytes, so the file is still a valid
# bytecode file for Python's import system.)

_MACRO_DEPS_MAGIC = b"\x00HYMACRODEPS1"
_MACRO_DEPS_LENGTH_BYTES = 4

# The modification time and size of each source file as of when we
# loaded it, so that a dependency is recorded as the version that's
# actually in memory.
_loaded_source_stats = {}

_py_path_stats = importlib.machinery.SourceFileLoader.path_stats


def _recording_path_stats(self, path):
    stats = _py_path_stats(self, path)
    _loaded_source_stats[path] = (stats["mtime"], stats["size"])
    return stats


importlib.machinery.SourceFileLoader.path_stats = _recording_path_stats


def _top_package(module_name):
    return module_name.partition(".")[0]


def _macro_providers(namespace):
    "Yield the loaded modules that define the macros in `namespace`."
    for table in ("_hy_macros", "_hy_reader_macros"):
        for macro in list((namespace.get(table) or {}).values()):
            provider = sys.modules.get(getattr(macro, "__module__", None))
            if provider is not None:
                yield provider


def _referenced_module_names(namespace):
    "Yield the names of the modules that the values in `namespace` come from."
    for value in list(namespace.values()):
        try:
            name = (
                value.__name__
                if inspect.ismodule(value)
                else getattr(value, "__module__", None))
        except Exception:
            continue
        if isinstance(name, str):
            yield name


def _macro_dependencies(module, path):
    """Return the source files that the expansion of `module` (compiled
    from `path`) depends on, as a sorted list of `[path, mtime, size]`.

    These are the files of the modules that provide `module`'s macros,
    plus the modules in the same top-level package that a provider
    refers to (a macro's helper functions), plus the same for the
    providers of each provider's own macros. Hy itself is covered by
    the Hy version in the record."""

    files = {}
    seen = set()
    pending = list(_macro_providers(vars(module)))
    while pending:
        provider = pending.pop()
        name = getattr(provider, "__name__", None)
        if not isinstance(name, str) or name in seen:
            continue
        seen.add(name)
        if provider is module or _top_package(name) == "hy":
            continue
        fname = getattr(provider, "__file__", None)
        if not isinstance(fname, str) or fname == path:
            continue
        if fname.endswith(tuple(importlib.machinery.SOURCE_SUFFIXES)):
            files[fname] = None
        pending.extend(_macro_providers(vars(provider)))
        for other in _referenced_module_names(vars(provider)):
            if other not in seen and _top_package(other) == _top_package(name):
                other = sys.modules.get(other)
                if other is not None:
                    pending.append(other)

    out = []
    for fname in sorted(files):
        stats = _loaded_source_stats.get(fname)
        if stats is None:
            try:
                st = os.stat(fname)
            except OSError:
                continue
            stats = (st.st_mtime, st.st_size)
        out.append([fname, *stats])
    return out


def _macro_deps_trailer(deps):
    record = json.dumps(dict(hy=hy.__version__, deps=deps)).encode("utf-8")
    return (
        record +
        len(record).to_bytes(_MACRO_DEPS_LENGTH_BYTES, "little") +
        _MACRO_DEPS_MAGIC)


def _macro_deps_are_current(bytecode):
    """Given the contents of a bytecode file, return true if it records
    its macro dependencies and they're all unchanged."""

    if not bytecode.endswith(_MACRO_DEPS_MAGIC):
        return False
    end = len(bytecode) - len(_MACRO_DEPS_MAGIC) - _MACRO_DEPS_LENGTH_BYTES
    if end < 0:
        return False
    length = int.from_bytes(
        bytecode[end : end + _MACRO_DEPS_LENGTH_BYTES], "little")
    if length > end:
        return False
    try:
        record = json.loads(bytes(bytecode[end - length : end]))
        if record["hy"] != hy.__version__:
            return False
        for fname, mtime, size in record["deps"]:
            st = os.stat(fname)
            if (st.st_mtime, st.st_size) != (mtime, size):
                return False
    except (ValueError, KeyError, TypeError, OSError):
        return False
    return True


_py_get_code = importlib.machinery.SourceFileLoader.get_code


def _hy_get_code(self, fullname):
    source_path = self.get_filename(fullname)
    if not _could_be_hy_src(source_path):
        return _py_get_code(self, fullname)
    try:
        bytecode_path = importlib.util.cache_from_source(source_path)
        bytecode = self.get_data(bytecode_path)
    except (NotImplementedError, OSError):
        # There's no bytecode to distrust. The usual path will compile
        # the source.
        return _py_get_code(self, fullname)
    if _macro_deps_are_current(bytecode):
        return _py_get_code(self, fullname)

    # Compile from source regardless of what the bytecode file says
    # about the source's modification time.
    try:
        stats = self.path_stats(source_path)
    except OSError:
        stats = None
    code = self.source_to_code(
        self.get_data(source_path),
        source_path,
        **(dict(fullname=fullname) if hy.compat.PY3_15 else {}))
    if stats is not None and not sys.dont_write_bytecode:
        try:
            self._cache_bytecode(
                source_path,
                bytecode_path,
                importlib._bootstrap_external._code_to_timestamp_pyc(
                    code, stats["mtime"], stats["size"]))
        except (NotImplementedError, OSError):
            pass
    return code


importlib.machinery.SourceFileLoader.get_code = _hy_get_code

_py_cache_bytecode = importlib.machinery.SourceFileLoader._cache_bytecode


def _hy_cache_bytecode(self, source_path, bytecode_path, data):
    deps = self.__dict__.pop("_hy_macro_deps", None)
    if deps is not None:
        data = bytes(data) + _macro_deps_trailer(deps)
    return _py_cache_bytecode(self, source_path, bytecode_path, data)


importlib.machinery.SourceFileLoader._cache_bytecode = _hy_cache_bytecode


def _hy_source_to_code(self, data, path, fullname=None, _optimize=-1):
    if _could_be_hy_src(path):
        if os.environ.get("HY_MESSAGE_WHEN_COMPILING"):
            print("Compiling", path, file=sys.stderr)
        source = data.decode("utf-8")
        hy_tree = read_many(source, filename=path, skip_shebang=True, reader=HyReader())
        with loader_module_obj(self) as module:
            data = hy_compile(hy_tree, module)
            self._hy_macro_deps = _macro_dependencies(module, path)

    return _py_source_to_code(
        self, data, path,
        _optimize=_optimize,
        **(dict(fullname=fullname) if hy.compat.PY3_15 else {}))


importlib.machinery.SourceFileLoader.source_to_code = _hy_source_to_code


if (".hy", False, False) not in zipimport._zip_searchorder:
    zipimport._zip_searchorder += ((".hy", False, False),)
    _py_compile_source = zipimport._compile_source

    def _hy_compile_source(pathname, source, module=None):
        if not pathname.endswith(".hy"):
            return _py_compile_source(
                pathname,
                source,
                *([module] if hy.compat.PY3_15 else []))
        mname = f"<zip:{pathname}>"
        sys.modules[mname] = types.ModuleType(mname)
        return compile(
            hy_compile(
                read_many(source.decode("UTF-8"), filename=pathname, skip_shebang=True, reader=HyReader()),
                sys.modules[mname],
            ),
            pathname,
            "exec",
            dont_inherit=True,
        )

    zipimport._compile_source = _hy_compile_source


#  This is actually needed; otherwise, pre-created finders assigned to the
#  current dir (i.e. `''`) in `sys.path` will not catch absolute imports of
#  directory-local modules!
sys.path_importer_cache.clear()

# Do this one just in case?
importlib.invalidate_caches()


class HyLoader(importlib.machinery.SourceFileLoader):
    pass


# We create a separate version of runpy, "runhy", that prefers Hy source over
# Python.
runhy = importlib.import_module("runpy")

runhy._get_code_from_file = partial(_get_code_from_file, hy_src_check=_could_be_hy_src)

del sys.modules["runpy"]

runpy = importlib.import_module("runpy")

_runpy_get_code_from_file = runpy._get_code_from_file
runpy._get_code_from_file = _get_code_from_file


def _inject_builtins():
    """Inject the Hy core macros into Python's builtins if necessary"""
    if hasattr(builtins, "__hy_injected__"):
        return
    hy.macros.load_macros(builtins)
    # Set the marker so we don't inject again.
    builtins.__hy_injected__ = True
