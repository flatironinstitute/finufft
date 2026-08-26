#!/bin/bash

# Assemble the report page from the sections the three halves stashed.
set -euxo pipefail

rm -f docs/pics/perftestci_*
mv outputs/*.svg docs/pics/
python3 tools/perftest-page.py \
	--page docs/performance.rst --snippets outputs
