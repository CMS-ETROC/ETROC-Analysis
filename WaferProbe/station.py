"""station.py -- helpers copied verbatim from the station repo
ETROC-WaferProbe, branch psu-identify at commit 949c043: the whole of
src/grading.py, plus nem_files (from src/qinj_check.py) and load_wafer_map
(from prober_move.py), and at commit 6327218 the quick test's pixels (from
master_run_script.py). Brought in so plot_wafer.py, wafer_tables.py,
wafer_plots.py and plot_lots.py can run here without the rest of the
station repo.

The station decides the grades, not this copy: when src/grading.py changes
on ETROC-WaferProbe, this file must be updated to follow it.
"""
import csv
import re
from collections import namedtuple
from pathlib import Path

# ---------------------------------------------------------------------------
# src/grading.py, whole file, verbatim (only its "import re" and
# "from collections import namedtuple" moved up to the top of this file).
# ---------------------------------------------------------------------------

"""grading.py -- turn a die's summary.json (written by
master_run_script.run_die) into a station bin number, a short grade name,
and a detail string for SetDieResult / BinMapDie.

The mapping is read off run_die's own code path, not guessed:

- run_die() marks "power_on", "i2c_start", "high_power", "qinj_start" and
  "power_off" into summary["phases"] as it goes (see the `mark()` calls),
  so which phase keys are present says which stage a "failed" run reached.
- the over-current check builds summary["short_check"][rail] = {"voltage",
  "current", "abort_above", "ok"} for every rail, before raising to set
  summary["status"] = "over_current" -- the rail name and its measured
  current are already there for any rail with ok == False, nothing new to
  add. The same check runs again once the chip is at high power, into
  summary["short_check_high"] (runs from 2026-09-26 on). The open-rail check that follows it adds "abort_below" for the
  rails with a yaml min_current and sets status "rail_open" when one
  draws less than that.
- summary["i2c"][chip_hex] = {"pixel_id": {...}, "peripheral": {...}} comes
  straight from src/i2cgui2_wrapper.py's pixel_check() and
  basic_peripheral_register_check(), each a {"passed": bool, ...} dict;
  batch_pixel_check()/batch_peripheral_check() never raise on a failed
  check, so a die with a failed check still finishes with status
  "completed".
- summary["calibration"]["zero_pixels"] lists the calibrated pixels whose
  baseline or noise width read 0. The calibration runs after the I2C
  checks and before QInj, and a zero reading does not stop the run, so a
  run that reached QInj also carries its verdict (summary["qinj_check"]).
  An imported run may add pixels its calibration did not read, carried
  over from another test by import_runs.py; they are listed again in
  summary["calibration"]["carried_over"] and counted apart.
- summary["efuse"][chip_hex] = {"word", "before", "after", "writes",
  "verified", "reason"} is filled in by src/efuse_check.burn_efuse during
  the eFuse burn (--doEfuse), or by src/efuse_check.verify_efuse, which
  reads the fuses and writes nothing to them (--verifyEfuse); either comes after
  the calibration and before QInj, and one that did not verify does not
  stop the run. Both are graded alike.
- summary["qinj_check"] is src/qinj_check.run_verdict's record. With
  --useEfuse after a burn or verify that verified it counts the events
  whose trailer carries another chip ID than the word's (chip_id_bad); a
  failed run with such events and none off the record pattern (bad)
  failed on the chip ID alone.
- a test still running at its time limit (src/die_watchdog.py) is stopped
  there and powered off, with status "stuck" and the limit and the last
  phase marked in summary["error"]; its I2C checks, calibration and eFuse
  records hold whatever the test had reached.
- a die the probe station could not step to (wafer_run.py: StepToDie
  refused with a code of CONTACT_FAILURE_CODES, e.g. 896 when Velox's
  pattern matching cannot find the trained wafer model at the die) never
  reaches the die test: wafer_run.py writes its summary.json itself, with
  status "contact_failure", the station's reply in summary["error"] and
  nothing else of a test.
"""


Grade = namedtuple("Grade", ["name", "bin", "detail"])

# Bin numbers, order and meaning fixed by hand -- must match the station's
# own bin-code table (see Prober.bin_code / wafer_run.start_checks).
PASSED = 0
POWER_SHORT = 1
I2C_NACK = 2
I2C_PIXELS = 3
NO_LINK_OR_DATA = 4
OTHER_FAIL = 5
NOT_TESTED = 6
RAIL_OPEN = 7
BL_NW_ZERO = 8
EFUSE_FAIL = 9
EFUSE_TRAILER_FAIL = 10
TEST_FAILURE = 11
CONTACT_FAILURE = 12
N_BINS = 13  # bins 0 .. N_BINS - 1, all read by wafer_run.start_checks

# SetDieResult: "Result (optional) -- No spaces (maximum 256 characters)"
# (Velox remote-interface manual, the SetDieResult entry).
_RESULT_TEXT_LIMIT = 256


def _shorted_rails(summary):
    """[(rail, current_amps, fault, suffix), ...] for every rail marked not
    ok in the power-on check (suffix "") or the high-power one ("_high"),
    in the yaml order."""
    out = []
    for key, suffix in (("short_check", ""), ("short_check_high", "_high")):
        for rail, info in (summary.get(key) or {}).items():
            if not info.get("ok", True) and info.get("fault") != "open":
                out.append((rail, info.get("current"), info.get("fault"), suffix))
    return out


def _open_rails(summary):
    """[(rail, current_amps), ...] for every rail whose short_check has an
    abort_below floor that its current did not reach."""
    return [(rail, info.get("current")) for rail, info in summary.get("short_check", {}).items()
            if "abort_below" in info and (info.get("current") or 0) < info["abort_below"]]


def _failed_checks(summary):
    """["<chip_hex>:pixel_id", "<chip_hex>:peripheral", ...] for every I2C
    check that did not pass."""
    failed = []
    for chip, checks in summary.get("i2c", {}).items():
        for check_name in ("pixel_id", "peripheral"):
            result = checks.get(check_name)
            if result and not result.get("passed", True):
                failed.append(f"{chip}:{check_name}")
    return failed


def zero_pixels(summary):
    """[[row, col], ...] of the calibrated pixels whose baseline or noise
    width read 0; empty when there are none or no calibration ran."""
    return (summary.get("calibration") or {}).get("zero_pixels") or []


def efuse_failures(summary):
    """[(chip_hex, reason), ...] for every chip whose eFuse burn, or
    read-only verify (verify_efuse), did not verify; empty when there are
    none or neither ran."""
    return [(chip, info.get("reason") or "not verified")
            for chip, info in (summary.get("efuse") or {}).items() if not info.get("verified")]


def _qinj_note(summary):
    """The QInj verdict as a detail suffix when the run reached the QInj
    stage, else ''."""
    if "qinj_start" not in summary.get("phases", {}):
        return ""
    check = summary.get("qinj_check") or {}
    if check.get("ok"):
        return f"; QInj OK, {check.get('events')} events"
    if summary.get("status") == "completed":
        return "; QInj ran, not checked"
    return f"; QInj failed: {check.get('reason') or summary.get('error') or 'no verdict'}"


def _before_qinj(summary):
    """The Grade of a finding of the stages before QInj -- a failed I2C
    check, then an eFuse burn that did not verify, then pixels reading BL
    or NW = 0 -- or None. The first finding in that order names the die,
    although the burn runs after the calibration; the QInj verdict goes
    into the detail."""
    failed = _failed_checks(summary)
    if failed:
        return Grade("I2C_PIXELS", I2C_PIXELS, ", ".join(failed) + _qinj_note(summary))
    unverified = efuse_failures(summary)
    if unverified:
        return Grade("EFUSE_FAIL", EFUSE_FAIL,
                     ", ".join(f"{chip}: {reason}" for chip, reason in unverified) + _qinj_note(summary))
    zero = zero_pixels(summary)
    if zero:
        calibration = summary.get("calibration") or {}
        carried = {tuple(p) for p in calibration.get("carried_over") or []}
        read = [p for p in zero if tuple(p) not in carried]
        n = calibration.get("n_pixels")
        more = len(zero) - len(read)
        if not read:
            count = f"{more} pixel{'s' if more > 1 else ''} BL or NW = 0, carried over from another test"
        else:
            count = (f"{len(read)} of {n} pixels" if n else f"{len(read)} pixels") + " BL or NW = 0"
            if more:
                count += f", {more} more carried over from another test"
        return Grade("BL_NW_ZERO", BL_NW_ZERO, count + _qinj_note(summary))
    return None


def _qinj_failure(summary, error):
    """The Grade of a run that failed in the QInj stage with no finding
    before it: EFUSE_TRAILER_FAIL when its check found trailers carrying
    another chip ID than the word's (--useEfuse) in events that all hold
    the record pattern, else NO_LINK_OR_DATA, which also takes a run with
    events off the pattern: their trailers are no more trustworthy than
    the rest."""
    check = summary.get("qinj_check") or {}
    if check.get("chip_id_bad") and not check.get("bad"):
        return Grade("EFUSE_TRAILER_FAIL", EFUSE_TRAILER_FAIL, error or "trailer chip ID")
    return Grade("NO_LINK_OR_DATA", NO_LINK_OR_DATA, error or "qinj stage failure")


def grade(summary):
    """The Grade for one die's summary.json. A missing summary (dry-run,
    or the die never ran) or one with no "status" field grades
    NOT_TESTED."""
    if not summary or "status" not in summary:
        return Grade("NOT_TESTED", NOT_TESTED, "no summary")

    status = summary.get("status")

    if status in (None, "dry_run", "not_run"):
        return Grade("NOT_TESTED", NOT_TESTED, "dry-run")

    if status == "over_current":
        shorted = _shorted_rails(summary)
        detail = "-".join(
            f"{rail}_{round((current or 0) * 1000)}mA" + ("_sag" if fault == "sag" else "") + suffix
            for rail, current, fault, suffix in shorted
        )
        return Grade("POWER_SHORT", POWER_SHORT, detail or "over-current")

    if status == "contact_failure":
        return Grade("CONTACT_FAILURE", CONTACT_FAILURE, summary.get("error") or "no contact")

    if status == "rail_open":
        opened = _open_rails(summary)
        detail = "-".join(
            f"{rail}_{round((current or 0) * 1000)}mA" for rail, current in opened
        )
        return Grade("RAIL_OPEN", RAIL_OPEN, detail or "no-current")

    if status == "completed":
        return _before_qinj(summary) or Grade("PASSED", PASSED, "")

    if status == "stuck":
        # The test ran past its time limit (src/die_watchdog.py) and was
        # stopped there; a finding made before it follows in the detail.
        detail = summary.get("error") or "stuck"
        first = _before_qinj(summary)
        if first:
            detail = f"{detail}; after {first.name}: {first.detail}"
        return Grade("TEST_FAILURE", TEST_FAILURE, detail)

    if status == "failed":
        phases = summary.get("phases", {})
        error = summary.get("error")
        if "qinj_start" in phases:
            return _before_qinj(summary) or _qinj_failure(summary, error)
        if "i2c_start" in phases:
            # a check or calibration that failed before the error still
            # names the die; the error follows in the detail
            first = _before_qinj(summary)
            if first:
                return first._replace(detail=f"{first.detail}; stopped: {error or 'i2c stage failure'}")
            return Grade("I2C_NACK", I2C_NACK, error or "i2c stage failure")
        return Grade("OTHER_FAIL", OTHER_FAIL, error or "failed before the i2c stage")

    # Any other status value (there should not be one) -- OTHER_FAIL is
    # the honest catch-all rather than a guess.
    return Grade("OTHER_FAIL", OTHER_FAIL, summary.get("error") or f"status {status!r}")


def result_text(grade, attempt):
    """The SetDieResult text for the die: "<GRADE>", or "<GRADE>_retry" when
    the graded attempt is the retry (attempt > 1). The grade's detail is
    recorded in the pass's wafer_<YYYYmmdd_HHMMSS>.json and the log, never
    on the map, so the station's Result column takes one of 2 x 13 values.
    Sanitised to [A-Za-z0-9_.-] and the station's 256-character limit all
    the same."""
    parts = [grade.name]
    if attempt > 1:
        parts.append("retry")
    text = "_".join(parts)
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:_RESULT_TEXT_LIMIT]


# ---------------------------------------------------------------------------
# src/qinj_check.py: nem_files only, verbatim.
# ---------------------------------------------------------------------------

def nem_files(run_dir):
    """The file_<n>.nem files of a run directory in numeric order (a plain
    sort would put file_10 before file_2)."""
    return sorted(Path(run_dir).glob("file_*.nem"),
                  key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


# ---------------------------------------------------------------------------
# prober_move.py: load_wafer_map only, verbatim.
# ---------------------------------------------------------------------------

def load_wafer_map(path):
    wafer_map = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            wafer_map[int(row["location_id"])] = (int(row["row"]), int(row["col"]))
    return wafer_map


# ---------------------------------------------------------------------------
# master_run_script.py: the pixels the quick test (no --doFullScan)
# calibrates, verbatim from run_die's pixels_of_interest.
# ---------------------------------------------------------------------------

QUICK_PIXELS = [(2, 2), (2, 10), (10, 2), (10, 10), (5, 5), (5, 13), (13, 5), (13, 13),
                (15, 15)]  # one pixel of the outermost ring
