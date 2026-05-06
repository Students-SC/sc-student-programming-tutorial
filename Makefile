SPHINXBUILD ?= .venv/bin/sphinx-build

.PHONY: docs docs-clean

docs:
	$(SPHINXBUILD) -b html . _build/html

docs-clean:
	rm -rf _build
