import ast
import importlib
import io
import marshal
import os
import runpy
import subprocess
import sys
import types
from importlib import reload
from pathlib import Path

import pytest

import hy
from hy.compiler import hy_compile
from hy.errors import HyLanguageError, hy_exc_handler
from hy.importer import HyLoader
from hy.reader import read_many


def test_basics():
    "Make sure the basics of the importer work"

    resources_mod = importlib.import_module("tests.resources")
    assert resources_mod.in_init == "chippy"

    bin_mod = importlib.import_module("tests.resources.bin")
    assert hasattr(bin_mod, "_null_fn_for_import_test")


def test_runpy():
    # `runpy` won't update cached bytecode. It's not clear if that's
    # intentional.

    basic_ns = runpy.run_path("tests/resources/importer/basic.hy")
    assert "square" in basic_ns

    main_ns = runpy.run_path("tests/resources/bin")
    assert main_ns["visited_main"] == 1
    del main_ns

    main_ns = runpy.run_module("tests.resources.bin")
    assert main_ns["visited_main"] == 1

    with pytest.raises(IOError):
        runpy.run_path("tests/resources/foobarbaz.py")


def test_stringer():
    _ast = hy_compile(
        read_many("(defn square [x] (* x x))"), __name__, import_stdlib=False
    )

    assert type(_ast.body[0]) == ast.FunctionDef


def test_imports():
    testLoader = HyLoader("tests.resources.importer.a", "tests/resources/importer/a.hy")
    spec = importlib.util.spec_from_loader(testLoader.name, testLoader)
    mod = importlib.util.module_from_spec(spec)

    with pytest.raises(NameError) as excinfo:
        testLoader.exec_module(mod)

    assert "thisshouldnotwork" in excinfo.value.args[0]


def test_import_error_reporting():
    "Make sure that (import) reports errors correctly."

    with pytest.raises(HyLanguageError):
        hy_compile(read_many('(import "sys")'), __name__)


def test_import_error_cleanup():
    "Failed initial imports should not leave dead modules in `sys.modules`."

    with pytest.raises(hy.errors.HyMacroExpansionError):
        importlib.import_module("tests.resources.fails")

    assert "tests.resources.fails" not in sys.modules


@pytest.mark.skipif(sys.dont_write_bytecode, reason="Bytecode generation is suppressed")
def test_import_autocompiles(tmp_path):
    "Test that (import) byte-compiles the module."

    p = tmp_path / "mymodule.hy"
    p.write_text('(defn pyctest [s] (+ "X" s "Y"))')

    def import_from_path(path):
        spec = importlib.util.spec_from_file_location("mymodule", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    assert import_from_path(p).pyctest("flim") == "XflimY"
    assert Path(importlib.util.cache_from_source(p)).exists()

    # Try running the bytecode.
    assert (
        import_from_path(importlib.util.cache_from_source(p)).pyctest("flam")
        == "XflamY"
    )


def test_eval():
    def eval_str(s):
        return hy.eval(hy.read(s))

    assert eval_str("[1 2 3]") == [1, 2, 3]
    assert eval_str('{"dog" "bark" "cat" "meow"}') == {"dog": "bark", "cat": "meow"}
    assert eval_str("#(1 2 3)") == (1, 2, 3)
    assert eval_str("#{3 1 2}") == {1, 2, 3}
    assert eval_str('(.strip " fooooo   ")') == "fooooo"
    assert (
        eval_str('(if True "this is if true" "this is if false")') == "this is if true"
    )
    assert eval_str("(lfor num (range 100) :if (= (% num 2) 1) (pow num 2))") == [
        pow(num, 2) for num in range(100) if num % 2 == 1
    ]


def test_reload(tmp_path, monkeypatch):
    """Generate a test module, confirm that it imports properly (and puts the
    module in `sys.modules`), then modify the module so that it produces an
    error when reloaded.  Next, fix the error, reload, and check that the
    module is updated and working fine.  Rinse, repeat.

    This test is adapted from CPython's `test_import.py`.
    """

    def unlink(filename):
        Path(source).unlink()
        bytecode = importlib.util.cache_from_source(source)
        if Path(bytecode).is_file():
            Path(bytecode).unlink()

    TESTFN = "testfn"
    source = tmp_path / (TESTFN + ".hy")
    source.write_text("(setv a 1)  (setv b 2)")

    monkeypatch.syspath_prepend(tmp_path)
    try:
        mod = importlib.import_module(TESTFN)
        assert TESTFN in sys.modules
        assert mod.a == 1
        assert mod.b == 2

        # On WinXP, just replacing the .py file wasn't enough to
        # convince reload() to reparse it.  Maybe the timestamp didn't
        # move enough.  We force it to get reparsed by removing the
        # compiled file too.
        unlink(source)

        # Now damage the module.
        source.write_text("(setv a 10)  (setv b (// 20 0))")

        with pytest.raises(ZeroDivisionError):
            reload(mod)

        # But we still expect the module to be in sys.modules.
        mod = sys.modules.get(TESTFN)
        assert mod is not None

        # We should have replaced a w/ 10, but the old b value should
        # stick.
        assert mod.a == 10
        assert mod.b == 2

        # Now fix the issue and reload the module.
        unlink(source)

        source.write_text("(setv a 11)  (setv b (// 20 1))")

        reload(mod)

        mod = sys.modules.get(TESTFN)
        assert mod is not None

        assert mod.a == 11
        assert mod.b == 20

        # Now cause a syntax error (a missing parenthesis)
        unlink(source)

        source.write_text("(setv a 11  (setv b (// 20 1))")

        with pytest.raises(hy.PrematureEndOfInput):
            reload(mod)

        mod = sys.modules.get(TESTFN)
        assert mod is not None

        assert mod.a == 11
        assert mod.b == 20

        # Fix it and retry
        unlink(source)

        source.write_text("(setv a 12)  (setv b (// 10 1))")

        reload(mod)

        mod = sys.modules.get(TESTFN)
        assert mod is not None

        assert mod.a == 12
        assert mod.b == 10

    finally:
        if TESTFN in sys.modules:
            del sys.modules[TESTFN]


def test_reload_reexecute(capsys):
    """A module is re-executed when it's reloaded, even if it's
    unchanged.

    https://github.com/hylang/hy/issues/712"""
    import tests.resources.hello_world

    assert capsys.readouterr().out == "hello world\n"
    assert capsys.readouterr().out == ""
    reload(tests.resources.hello_world)
    assert capsys.readouterr().out == "hello world\n"


def test_circular(monkeypatch):
    """Test circular imports by creating a temporary file/module that calls a
    function that imports itself."""
    monkeypatch.syspath_prepend("tests/resources/importer")
    assert runpy.run_module("circular")["f"]() == 1


def test_shadowed_basename(monkeypatch):
    """Make sure Hy loads `.hy` files instead of their `.py` counterparts (.e.g
    `__init__.py` and `__init__.hy`).
    """
    monkeypatch.syspath_prepend("tests/resources/importer")
    foo = importlib.import_module("foo")
    assert Path(foo.__file__).name == "__init__.hy"
    assert foo.ext == "hy"
    some_mod = importlib.import_module("foo.some_mod")
    assert Path(some_mod.__file__).name == "some_mod.hy"
    assert some_mod.ext == "hy"


def test_docstring(monkeypatch):
    """Make sure a module's docstring is loaded."""
    monkeypatch.syspath_prepend("tests/resources/importer")
    mod = importlib.import_module("docstring")
    expected_doc = "This module has a docstring.\n\n" "It covers multiple lines, too!\n"
    assert mod.__doc__ == expected_doc
    assert mod.a == 1


def test_hy_python_require():
    # https://github.com/hylang/hy/issues/1911
    test = "(do (require tests.resources.macros [test-macro]) (test-macro) blah)"
    assert hy.eval(hy.read(test)) == 1


def test_filtered_importlib_frames(capsys):
    testLoader = HyLoader(
        "tests.resources.importer.compiler_error",
        "tests/resources/importer/compiler_error.hy",
    )
    spec = importlib.util.spec_from_loader(testLoader.name, testLoader)
    mod = importlib.util.module_from_spec(spec)

    with pytest.raises(hy.PrematureEndOfInput) as execinfo:
        testLoader.exec_module(mod)

    hy_exc_handler(execinfo.type, execinfo.value, execinfo.tb)
    captured_w_filtering = capsys.readouterr()[-1].strip()

    assert "importlib._" not in captured_w_filtering


def test_zipimport(tmp_path, monkeypatch):
    from zipfile import ZipFile

    zpath = tmp_path / "archive.zip"
    with ZipFile(zpath, "w") as o:
        # Test a Python module as well as a Hy module to ensure that
        # Hy's edits to ZIP imports can still get Python files.
        o.writestr("zip_py.py", 'x = "Py from ZIP"')
        o.writestr("zip_hy.hy", '(setv x "Hy from ZIP")')

    monkeypatch.syspath_prepend(zpath)
    import zip_py, zip_hy
    assert zip_py.x == "Py from ZIP"
    assert zip_py.__file__ == str(zpath / "zip_py.py")
    assert zip_hy.x == "Hy from ZIP"
    assert zip_hy.__file__ == str(zpath / "zip_hy.hy")


def test_eval_requiring_macro():
    # https://github.com/hylang/hy/issues/2695
    hy.eval(hy.read("(require tests.resources.macros)"), globals={})


class MacroDepsProject:
    """A package `pkg` with a macro module, a helper module that the
    macro calls, and a user module, imported in fresh interpreters."""

    def __init__(self, root):
        self.root = root
        self.pkg = root / "pkg"
        self.pkg.mkdir()
        (self.pkg / "__init__.py").write_text("")
        self.helper = self.pkg / "helper.py"
        self.macros = self.pkg / "macros.hy"
        self.user = root / "user.hy"
        self.write(self.helper, 'def suffix(): return "h1"')
        self.write(self.macros, self.macro_source("m1"))
        self.write(self.user, "(require pkg.macros [m]) (setv x (m))")

    @staticmethod
    def macro_source(tag):
        return f'(import pkg.helper [suffix]) (defmacro m [] (+ "{tag}-" (suffix)))'

    @staticmethod
    def write(path, text):
        "Write `text` and move the modification time forward, as an edit would."
        previous = path.stat().st_mtime_ns if path.exists() else None
        path.write_text(text)
        if previous is not None:
            later = previous + 10 * 10**9
            os.utime(path, ns=(later, later))

    def bytecode(self, source):
        return Path(importlib.util.cache_from_source(str(source)))

    def record(self, source):
        from hy.importer import _macro_deps_path
        return Path(_macro_deps_path(str(self.bytecode(source))))

    def run(self):
        "Import `user` in a new interpreter. Return `x` and the files compiled."
        result = subprocess.run(
            [sys.executable, "-c", "import hy, user; print(user.x)"],
            cwd=self.root,
            env={
                **os.environ,
                "PYTHONPATH": str(self.root),
                "HY_MESSAGE_WHEN_COMPILING": "1",
                "PYTHONDONTWRITEBYTECODE": "",
                "PYTHONPYCACHEPREFIX": ""},
            capture_output=True,
            text=True,
            check=True)
        compiled = {
            Path(line.removeprefix("Compiling ")).name
            for line in result.stderr.splitlines()
            if line.startswith("Compiling ")}
        return result.stdout.strip(), compiled


bytecode_is_written = pytest.mark.skipif(
    sys.dont_write_bytecode or bool(sys.pycache_prefix),
    reason="Bytecode isn't written beside the source")


@bytecode_is_written
def test_macro_deps_unchanged(tmp_path):
    "Bytecode is reused so long as the macros it was expanded with are unchanged."
    project = MacroDepsProject(tmp_path)
    assert project.run() == ("m1-h1", {"macros.hy", "user.hy"})
    assert project.run() == ("m1-h1", set())


@bytecode_is_written
def test_macro_deps_macro_changed(tmp_path):
    "A change to a macro recompiles the modules that use it."
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    project.write(project.macros, project.macro_source("m2"))
    assert project.run() == ("m2-h1", {"macros.hy", "user.hy"})
    assert project.run() == ("m2-h1", set())


@bytecode_is_written
def test_macro_deps_helper_changed(tmp_path):
    """A change to a module that a macro's module uses, in the same
    package, recompiles the modules that use the macro."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    project.write(project.helper, 'def suffix(): return "h2"')
    assert project.run() == ("m1-h2", {"user.hy"})
    assert project.run() == ("m1-h2", set())


@bytecode_is_written
def test_macro_deps_transitive(tmp_path):
    """A change to the macro that a macro was written with recompiles
    the users of the latter."""
    project = MacroDepsProject(tmp_path)
    base = project.pkg / "base.hy"
    project.write(base, '(defmacro tag [] "b1")')
    project.write(
        project.macros,
        "(require pkg.base [tag]) (defmacro m [] (tag))")
    assert project.run()[0] == "b1"
    project.write(base, '(defmacro tag [] "b2")')
    assert project.run() == ("b2", {"base.hy", "macros.hy", "user.hy"})
    assert project.run() == ("b2", set())


def pyc_flags(data):
    return int.from_bytes(data[4:8], "little")


def assert_regular_bytecode(data):
    "`data` is a bytecode file that `marshal` reads to the end."
    assert data[:4] == importlib.util.MAGIC_NUMBER
    f = io.BytesIO(data[16:])
    assert isinstance(marshal.load(f), types.CodeType)
    assert f.tell() == len(data) - 16


@bytecode_is_written
def test_macro_deps_regular_bytecode(tmp_path):
    """The record is written beside the bytecode file, which is a
    regular bytecode file with nothing after the marshalled code."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    for source in (project.user, project.macros):
        bytecode = project.bytecode(source)
        assert_regular_bytecode(bytecode.read_bytes())
        assert project.record(source).exists()


@bytecode_is_written
@pytest.mark.parametrize("damage", ["delete", "garble"])
def test_macro_deps_unrecorded_bytecode(tmp_path, damage):
    """Bytecode with no usable record of its macros (e.g., from an older
    Hy) is recompiled."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    record = project.record(project.user)
    original = record.read_bytes()

    if damage == "delete":
        record.unlink()
    else:
        record.write_bytes(b"{")

    assert project.run() == ("m1-h1", {"user.hy"})
    assert record.read_bytes() == original
    assert project.run() == ("m1-h1", set())


@bytecode_is_written
def test_macro_deps_record_of_other_source(tmp_path):
    """A record left from bytecode compiled from another version of the
    source (e.g., when an older Hy recompiled the module after an edit)
    doesn't vouch for the current bytecode."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    record = project.record(project.user)
    stale = record.read_bytes()
    project.write(project.user, '(require pkg.macros [m]) (setv x (+ (m) "!"))')
    assert project.run() == ("m1-h1!", {"user.hy"})
    current = record.read_bytes()

    record.write_bytes(stale)
    assert project.run() == ("m1-h1!", {"user.hy"})
    assert record.read_bytes() == current
    assert project.run() == ("m1-h1!", set())


@bytecode_is_written
def test_macro_deps_remarshalled_bytecode(tmp_path):
    """Rewriting only how the code is marshalled, as a build step may do
    for reproducibility, keeps the record valid."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    bytecode = project.bytecode(project.user)
    data = bytecode.read_bytes()
    rewritten = data[:16] + marshal.dumps(marshal.loads(data[16:]), 2)
    assert rewritten != data
    bytecode.write_bytes(rewritten)

    assert project.run() == ("m1-h1", set())
    assert bytecode.read_bytes() == rewritten


@bytecode_is_written
def test_macro_deps_by_contents(tmp_path):
    """A dependency is compared by its contents, so a new modification
    time alone doesn't recompile the modules that use its macros."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    project.write(project.helper, project.helper.read_text())
    project.write(project.macros, project.macros.read_text())
    # `macros.hy` is recompiled because Python checks its bytecode
    # against its own modification time, but `user.hy` isn't.
    assert project.run() == ("m1-h1", {"macros.hy"})
    assert project.run() == ("m1-h1", set())


def compile_hash_based(source, mode):
    """Compile `source` to a hash-based bytecode file with `py_compile`,
    in a new interpreter, as a build step would."""
    subprocess.run(
        [sys.executable, "-c",
            "import sys, hy, py_compile; "
            "py_compile.compile(sys.argv[1], doraise=True, "
            f"invalidation_mode=py_compile.PycInvalidationMode.{mode})",
            str(source)],
        cwd=source.parent,
        env={**os.environ, "PYTHONPATH": str(source.parent)},
        check=True)


@bytecode_is_written
def test_macro_deps_unchecked_hash(tmp_path):
    """Bytecode that Python uses without checking its source is also
    used without checking its macros, with or without a record."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    for source in (project.user, project.macros):
        project.record(source).unlink()
        compile_hash_based(source, "UNCHECKED_HASH")
    user_bytecode = project.bytecode(project.user).read_bytes()
    assert pyc_flags(user_bytecode) == 0b01

    assert project.run() == ("m1-h1", set())
    project.write(project.helper, 'def suffix(): return "h2"')
    assert project.run() == ("m1-h1", set())
    assert project.bytecode(project.user).read_bytes() == user_bytecode
    assert not project.record(project.user).exists()


@bytecode_is_written
def test_macro_deps_checked_hash(tmp_path):
    """Bytecode that Python checks by the hash of its source is checked
    against its macros too, and is recompiled to the same kind."""
    project = MacroDepsProject(tmp_path)
    assert project.run()[0] == "m1-h1"
    compile_hash_based(project.user, "CHECKED_HASH")
    project.record(project.user).unlink()

    assert project.run() == ("m1-h1", {"user.hy"})
    assert pyc_flags(project.bytecode(project.user).read_bytes()) == 0b11
    assert project.run() == ("m1-h1", set())
    project.write(project.helper, 'def suffix(): return "h2"')
    assert project.run() == ("m1-h2", {"user.hy"})
    assert pyc_flags(project.bytecode(project.user).read_bytes()) == 0b11
    assert project.run() == ("m1-h2", set())
