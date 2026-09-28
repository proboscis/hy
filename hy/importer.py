import _imp
import builtins
import hashlib
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
# for a bytecode file looks only at the source it was compiled from, so
# after a change to a macro, modules that use the macro would keep
# running their old expansions. To prevent this, we write beside each
# bytecode file a record of the files that the module's macros came
# from, and recompile when the record is missing or no longer matches.
# The bytecode file itself stays a regular bytecode file, so that other
# tools that read or rewrite bytecode files keep working.
#
# The record is checked only when Python itself would check the
# bytecode file against its source. In particular, a hash-based
# bytecode file that Python doesn't check (PEP 552's "unchecked-hash")
# is trusted as it is, and so is its expansion of macros.

_MACRO_DEPS_SUFFIX = ".hydeps"

# The digest of each source file's contents, computed at most once per
# process. A module compiled in this process has the digest of the
# source that was actually compiled, so that a dependency is recorded as
# the version that's in memory.
_source_digests = {}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _source_digest(path):
    if path not in _source_digests:
        with open(path, "rb") as o:
            _source_digests[path] = _digest(o.read())
    return _source_digests[path]


def _macro_deps_path(bytecode_path):
    return os.path.splitext(bytecode_path)[0] + _MACRO_DEPS_SUFFIX


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
    from `path`) depends on, as a sorted list of `[path, digest]`.

    These are the files of the modules that provide `module`'s macros,
    plus the modules in the same top-level package that a provider
    refers to (a macro's helper functions), plus the same for the
    providers of each provider's own macros. Hy itself is covered by
    the Hy version in the record."""

    files = set()
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
            files.add(fname)
        pending.extend(_macro_providers(vars(provider)))
        for other in _referenced_module_names(vars(provider)):
            if other not in seen and _top_package(other) == _top_package(name):
                other = sys.modules.get(other)
                if other is not None:
                    pending.append(other)

    out = []
    for fname in sorted(files):
        try:
            out.append([fname, _source_digest(fname)])
        except OSError:
            continue
    return out


def _macro_deps_record(bytecode, deps):
    "Return the record for the bytecode file `bytecode`."
    return json.dumps(dict(
        hy=hy.__version__,
        bytecode=_digest(bytecode),
        deps=deps)).encode("utf-8")


def _macro_deps_are_current(record, bytecode):
    """Given the contents of a record and of the bytecode file beside
    it, return true if the record is of that bytecode file and its
    macro dependencies are all unchanged."""

    try:
        record = json.loads(record)
        if (record["hy"] != hy.__version__ or
                record["bytecode"] != _digest(bytecode)):
            return False
        for fname, digest in record["deps"]:
            if _source_digest(fname) != digest:
                return False
    except (ValueError, KeyError, TypeError, OSError):
        return False
    return True


def _python_checks_source(bytecode):
    """Return true if Python checks the bytecode file `bytecode` against
    its source before using it (as for the default timestamp-based
    files and PEP 552's "checked-hash" files)."""
    flags = int.from_bytes(bytecode[4:8], "little")
    if not flags & 0b01:
        return True
    mode = _imp.check_hash_based_pycs
    return mode == "always" or (mode != "never" and bool(flags & 0b10))


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
    if (len(bytecode) < 16 or
            bytecode[:4] != importlib.util.MAGIC_NUMBER or
            not _python_checks_source(bytecode)):
        # Python will either recompile the source anyway or use the
        # bytecode without checking it.
        return _py_get_code(self, fullname)
    try:
        if _macro_deps_are_current(
                self.get_data(_macro_deps_path(bytecode_path)), bytecode):
            return _py_get_code(self, fullname)
    except OSError:
        pass

    # Compile from source regardless of what the bytecode file says
    # about the source, and write the same kind of bytecode file.
    source = self.get_data(source_path)
    try:
        stats = self.path_stats(source_path)
    except OSError:
        stats = None
    code = self.source_to_code(
        source,
        source_path,
        **(dict(fullname=fullname) if hy.compat.PY3_15 else {}))
    if stats is not None and not sys.dont_write_bytecode:
        flags = int.from_bytes(bytecode[4:8], "little")
        data = (
            importlib._bootstrap_external._code_to_hash_pyc(
                code,
                importlib.util.source_hash(source),
                bool(flags & 0b10))
            if flags & 0b01 else
            importlib._bootstrap_external._code_to_timestamp_pyc(
                code, stats["mtime"], stats["size"]))
        try:
            self._cache_bytecode(source_path, bytecode_path, data)
        except (NotImplementedError, OSError):
            pass
    return code


importlib.machinery.SourceFileLoader.get_code = _hy_get_code

_py_cache_bytecode = importlib.machinery.SourceFileLoader._cache_bytecode


def _hy_cache_bytecode(self, source_path, bytecode_path, data):
    deps = self.__dict__.pop("_hy_macro_deps", None)
    result = _py_cache_bytecode(self, source_path, bytecode_path, data)
    if deps is not None:
        # The record names the bytecode it describes, so if only one of
        # the two files gets written, the pair is recompiled next time.
        self.set_data(
            _macro_deps_path(bytecode_path),
            _macro_deps_record(bytes(data), deps))
    return result


importlib.machinery.SourceFileLoader._cache_bytecode = _hy_cache_bytecode


def _hy_source_to_code(self, data, path, fullname=None, _optimize=-1):
    if _could_be_hy_src(path):
        if os.environ.get("HY_MESSAGE_WHEN_COMPILING"):
            print("Compiling", path, file=sys.stderr)
        source = data.decode("utf-8")
        hy_tree = read_many(source, filename=path, skip_shebang=True, reader=HyReader())
        _source_digests[path] = _digest(data)
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
