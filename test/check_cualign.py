"""Integration checks for built cualign and optionally cualign.cuda."""
from pathlib import Path
import subprocess
import sys
import tempfile


def graph(path, names, arcs):
    indices = {name: i for i, name in enumerate(names, 1)}
    rows = ["LEDA.GRAPH", "string", "short", "-2", str(len(names))]
    rows.extend(f"|{{{name}}}|" for name in names)
    rows.append(str(len(arcs)))
    rows.extend(f"{indices[u]} {indices[v]} 0 |{{}}|" for u, v in arcs)
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def check(binaries, root, name, source, target, arcs_a, arcs_b, rows,
          score=None, factors=None, error=None, extra=()):
    a, b, candidates = (root / f"{name}.{suffix}" for suffix in ("a.gw", "b.gw", "candidates"))
    graph(a, source, arcs_a)
    graph(b, target, arcs_b)
    candidates.write_text(rows, encoding="ascii")
    maps = []
    for binary, backend in binaries:
        output = root / f"{name}.{backend}.align"
        result = subprocess.run([
            binary, "--g1", a, "--g2", b, "--candidates", candidates,
            "--output", output, "--backend", backend, "--iterations", "6", *extra,
        ], text=True, capture_output=True, timeout=30)
        if error:
            assert result.returncode and error in result.stderr, result
            assert not output.exists()
            continue
        assert result.returncode == 0, result
        line = next(line for line in result.stdout.splitlines() if line.startswith("cualign,"))
        stats = dict(entry.split("=", 1) for entry in line.split(",")[1:])
        pairs = [line.split() for line in output.read_text().splitlines()]
        mapping = dict(pairs)
        allowed = set()
        for row in rows.splitlines():
            src, *targets = row.split()
            allowed.update((src, t) for t in targets)
        assert len(pairs) == len(source) and set(mapping) == set(source)
        assert len(set(mapping.values())) == len(source) and set(mapping.values()) <= set(target)
        assert set(mapping.items()) <= allowed
        exact = sum((mapping[u], mapping[v]) in arcs_b for u, v in arcs_a)
        assert exact == int(stats["conserved"]) == score, result
        if factors is not None:
            assert int(stats["factors"]) == factors, result
        maps.append(mapping)
    if maps:
        assert all(mapping == maps[0] for mapping in maps), maps


def main():
    binaries = [(Path(sys.argv[1]).resolve(), "cpu")]
    if len(sys.argv) > 2:
        binaries.append((Path(sys.argv[2]).resolve(), "verify"))
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        check(binaries, root, "known", "abc", "wxyz",
              {("a", "b"), ("b", "a"), ("b", "c"), ("c", "c")},
              {("x", "y"), ("y", "x"), ("y", "z"), ("z", "z")},
              "a w x y z\nb w x y z\nc w x y z\n", score=4)
        check(binaries, root, "reciprocal", "ab", "xyz",
              {("a", "b"), ("b", "a")}, {("x", "y"), ("y", "x")},
              "a x\nb y\n", score=2, factors=1)
        check(binaries, root, "reverse", "ab", "xyz",
              {("a", "b")}, {("y", "x")}, "a x\nb y\n", score=0, factors=0)
        check(binaries, root, "repair", "ab", "xyz", {("a", "a")}, {("x", "x")},
              "a x y\nb x\n", score=0, factors=0)
        check(binaries, root, "hall", "abc", "xyz", set(), set(),
              "a x\nb x\nc y z\n", error="no full injection")
        check(binaries, root, "bounded", "ab", "xyz", {("a", "b")}, {("x", "y")},
              "a x\nb y\n", error="Overlap factor limit", extra=("--max-factors", "0"))
        bad_graph = root / "bad.gw"
        bad_graph.write_text("LEDA.GRAPH\nstring\nshort\n-2\n1\n|{a}|\n1\n0 1 0 |{}|\n")
        for binary, backend in binaries:
            output = root / f"bad.{backend}.align"
            result = subprocess.run([
                binary, "--g1", bad_graph, "--g2", root / "known.b.gw",
                "--candidates", root / "known.candidates", "--output", output,
                "--backend", backend,
            ], text=True, capture_output=True, timeout=30)
            assert result.returncode and "Invalid GW arc endpoint" in result.stderr, result
            assert not output.exists()
    print("cuAlign CLI checks passed")


if __name__ == "__main__":
    main()
