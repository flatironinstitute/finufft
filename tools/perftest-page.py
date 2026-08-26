"""Assemble docs/performance.rst from the measured sections.

``run_perftest_ci.py`` renders one ``performance_<backend>.rst`` and its
figures per backend. This stitches them under the marker in the in-tree
page, keeping the marker so the next run can find it again.

The two libraries are the page's two main sections, so a reader looking for
one of them navigates to it rather than scrolling past the other.
"""

import argparse
import sys
from pathlib import Path

MARKER = ".. PERFTEST_BACKENDS_BELOW"
# One entry per main section: its heading, then the backends under it in the
# order the page reads in. The GPU library is one backend, the CPU library has
# a choice of FFT libraries.
SECTIONS = (("CPU", ("fftw", "ducc")), ("GPU", ("cuda",)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--snippets", type=Path, required=True)
    args = parser.parse_args()

    intro, _, _ = args.page.read_text(encoding="utf-8").partition(MARKER)
    chunks = [intro.rstrip() + "\n\n" + MARKER + "\n"]
    for heading, backends in SECTIONS:
        found = [
            (args.snippets / f"performance_{backend}.rst").read_text(encoding="utf-8")
            for backend in backends
            if (args.snippets / f"performance_{backend}.rst").exists()
        ]
        # A half that failed leaves its section out rather than the page: the
        # other half is still worth publishing. Silence would read as coverage.
        if not found:
            print(f"no snippet for the {heading} section", file=sys.stderr)
            continue
        chunks.append(f"\n{heading}\n{'-' * len(heading)}\n")
        chunks += ["\n" + snippet.strip() + "\n" for snippet in found]
    args.page.write_text("\n".join(chunks) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
