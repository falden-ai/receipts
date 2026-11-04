FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

# Build-time egress: Debian apt mirrors and PyPI only. The source enters as a pinned local archive.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && dpkg -l > /usr/local/share/receipts-demo-dpkg.txt \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements.lock
COPY pandas-82315e7.with-whatsnew.tar /tmp/pandas-source.tar
RUN mkdir /src /workspace && tar -xf /tmp/pandas-source.tar -C /src \
    && cp -a /src/. /workspace/ \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        --config-settings=compile-args=-j2 /src

# Copy the non-editable wheel's extensions and generated files, preserving the base Python source and tests. The
# script is written out with printf, never a heredoc: the eval VM's docker has no buildx, and its legacy builder runs
# a RUN heredoc with an empty body, so the copy would never run there (FAL-643). The wheel's pandas/pyproject.toml
# (the repository's root config, installed for pandas.test()) is not copied: in /workspace/pandas it would become
# pytest's rootdir, and every case id would lose its pandas/ prefix, so a replay never finds its case (FAL-643).
RUN printf '%s\n' \
        'import shutil' \
        'import sysconfig' \
        'from pathlib import Path' \
        '' \
        'installed = Path(sysconfig.get_paths()["platlib"]) / "pandas"' \
        'tree = Path("/workspace/pandas")' \
        'extensions = 0' \
        'for source in installed.rglob("*"):' \
        '    if not source.is_file() or "__pycache__" in source.parts or source == installed / "pyproject.toml":' \
        '        continue' \
        '    destination = tree / source.relative_to(installed)' \
        '    if source.suffix == ".so" or not destination.exists():' \
        '        destination.parent.mkdir(parents=True, exist_ok=True)' \
        '        shutil.copy2(source, destination)' \
        '        extensions += source.suffix == ".so"' \
        'assert extensions > 0' \
        'assert (tree / "_version_meson.py").is_file()' \
        'assert not (tree / "pyproject.toml").exists()' \
        > /tmp/copy-extensions.py \
    && python /tmp/copy-extensions.py && rm /tmp/copy-extensions.py
# The workspace alone must import: pandas from /workspace, with the wheel uninstalled and no bytecode written. And
# pytest's rootdir must be /workspace, so a case id is pandas/tests/...: the collected ids must say so.
RUN python -m pip uninstall -y pandas && rm -rf /src /tmp/pandas-source.tar /tmp/requirements.lock \
    && test "$(du -sb /workspace | cut -f1)" -lt 700000000 \
    && cd / && PYTHONPATH=/workspace python -B -c \
        "import pandas; assert pandas.__file__ == '/workspace/pandas/__init__.py', pandas.__file__" \
    && cd /workspace && PYTHONPATH=/workspace python -B -m pytest -p no:cacheprovider --collect-only -q \
        pandas/tests/groupby/test_api.py | grep -q '^pandas/tests/groupby/test_api.py::'

ENV PYTHONPATH=/workspace
WORKDIR /workspace
