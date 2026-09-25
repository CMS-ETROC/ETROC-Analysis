"""Input check shared by every notebook.

A campaign keeps its tables in one folder outside the repository (the campaign module's INPUTS)
and lists every file there with its md5 in a manifest (INPUTS_MANIFEST, md5sum format, paths
relative to INPUTS). Files read in place elsewhere (raw logs, say) are checked for existence and
read permission only. Each notebook checks the tables it reads before drawing anything.
"""
import hashlib
import os

from .campaigns import active as _campaign


def _read_problem(path):
    """None if `path` (a file or a folder) can be read, else "missing" or "unreadable (why)"."""
    try:
        if os.path.isdir(path):
            os.listdir(path)
        else:
            with open(path, "rb") as fh:
                fh.read(1)
    except (FileNotFoundError, NotADirectoryError):
        return "missing"
    except OSError as err:
        return "unreadable (%s)" % (err.strerror or err)
    return None


def check_inputs(inputs=None, manifest=None, raw_inputs=(), only=None):
    """Check the inputs a notebook reads before any figure is drawn.

    Two sets: the tables in `inputs`, against the checksum list `manifest`, and the files in
    `raw_inputs`, read in place. `only` limits the tables to the manifest entries under the given
    folders or files (paths relative to `inputs`, such as "tables/" or "july/good_runs.csv");
    None checks every entry. Every file must exist and be readable; all problems are collected,
    missing apart from unreadable (no permission), and raised together. A table whose content
    differs from the published one (a cache you rebuilt, say) is reported and the run goes on,
    but the figures drawn from it will differ. `inputs` and `manifest` default to the campaign's
    INPUTS and INPUTS_MANIFEST.
    """
    inputs = _campaign.INPUTS if inputs is None else inputs
    manifest = _campaign.INPUTS_MANIFEST if manifest is None else manifest
    with open(manifest) as fh:
        listed = [line.split(None, 1) for line in fh if line.strip()]
    listed = [(md5, name.strip()) for md5, name in listed]
    if only is not None:
        listed = [(md5, name) for md5, name in listed
                  if any(name == o or (o.endswith("/") and name.startswith(o)) for o in only)]
        if not listed:
            raise RuntimeError("no entry of %s is under %s" % (manifest, ", ".join(only)))
    missing, unreadable, changed = [], [], []

    def note(path, problem):
        if problem == "missing":
            missing.append(path)
        else:
            unreadable.append("%s: %s" % (path, problem))

    for md5, name in listed:
        path = os.path.join(inputs, name)
        problem = _read_problem(path)
        if problem:
            note(path, problem)
            continue
        digest = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
        except OSError as err:
            note(path, "unreadable (%s)" % (err.strerror or err))
            continue
        if digest.hexdigest() != md5:
            changed.append(name)
    for path in raw_inputs:
        problem = _read_problem(path)
        if problem:
            note(path, problem)
    if missing or unreadable:
        raise RuntimeError(
            "input check failed for campaign %s (tables folder %s, %d tables and %d raw inputs "
            "checked):\n%s\nMissing: point the campaign's input variables at a full copy; "
            "campaigns/%s_inputs.md lists every variable and its default. "
            "Unreadable: ask the owner of that area for read access."
            % (_campaign.__name__, inputs, len(listed), len(raw_inputs),
               "\n".join(["  missing: %s" % p for p in missing]
                         + ["  unreadable: %s" % p for p in unreadable]),
               _campaign.__name__.split(".")[-1]))
    for name in changed:
        print("note: %s differs from the published file, so figures drawn from it will too"
              % name)
