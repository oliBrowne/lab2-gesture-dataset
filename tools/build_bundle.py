"""Pure-Python (no AI, no execution, no pip packages) -- builds a plain-
text "bundle" summarizing your dataset submission, for checking your own
work against RUBRIC.md before you submit it.

Usage:
    python3 build_bundle.py /path/to/your-submission --out bundle.txt

Then take bundle.txt plus RUBRIC.md to any LLM chat (Claude.ai, ChatGPT,
etc.) and ask it to grade your submission against the rubric -- see
README-student.md's "Before you submit: grade yourself" for a ready-to-
paste prompt.

What this does: walks your submission directory, extracts your README,
computes basic file/CSV statistics, guesses at your label scheme from
directory structure, and collects the full source of every .py/.ipynb
file found -- unfiltered by any guess about which one is "the"
augmentation script or "the" demo script (the LLM you paste this into
figures that out itself, by reading your README alongside the code, the
same way a human grader would).

.ipynb notebooks are rendered as their code+markdown cells, in order,
with stored outputs/execution counts/metadata stripped out -- those
reflect whoever last ran the notebook (possibly stale, or huge if a
cell's output is an embedded plot image), not the logic being graded.

Nothing here ever executes your code, and nothing here sends anything
anywhere over the network -- this only reads local files and writes
bundle.txt. Needs nothing beyond a standard Python 3 install.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

MAX_FILES_LISTED = 200
MAX_CSV_FILES_SUMMARIZED = 40
MAX_README_CHARS = 8000
MAX_TOTAL_SOURCE_CHARS = 25000  # combined budget across all .py/.ipynb files
MAX_SINGLE_SOURCE_CHARS = 12000  # cap per file within that budget
MAX_DEPENDENCY_MANIFEST_CHARS = 4000  # combined budget for all requirements*.txt files

# Hard backstop on the ENTIRE rendered bundle, regardless of how the
# per-section budgets above combine -- keeps what you paste into an LLM
# chat to a reasonable size no matter how large your submission is.
MAX_BUNDLE_CHARS = 60000

SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "env", ".pytest_cache", "node_modules",
    ".ipynb_checkpoints",
}

AUGMENTATION_KEYWORDS = ["augment", "synthesize", "jitter"]
DEMO_KEYWORDS = ["train", "demo", "model", "classif"]


@dataclass
class ScriptSource:
    rel_path: str
    source_text: str
    truncated: bool


@dataclass
class Analysis:
    file_list: list[str]
    total_file_count: int
    readme_text: str | None
    readme_path: str | None
    csv_summaries: list[str]
    label_guess: str
    script_sources: list[ScriptSource]
    script_sources_omitted_count: int
    dependency_manifests: list[ScriptSource]


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        yield p


def _find_readme(root: Path) -> Path | None:
    candidates = [p for p in root.glob("README*") if p.is_file()]
    if not candidates:
        candidates = [p for p in root.rglob("README*") if p.is_file() and not any(part in SKIP_DIR_NAMES for part in p.parts)]
    if not candidates:
        return None
    # Prefer README.md, then shortest path (closest to root).
    candidates.sort(key=lambda p: (p.name != "README.md", len(p.parts)))
    return candidates[0]


def _summarize_csv(path: Path) -> str:
    try:
        with open(path, newline="", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"{path}: ERROR reading file ({e})"

    # Skip leading '#'-prefixed comment lines (a common capture-tool convention).
    data_lines = [l for l in lines if not l.startswith("#")]
    if not data_lines:
        return f"{path}: empty"
    try:
        reader = csv.reader(data_lines)
        rows = list(reader)
    except Exception as e:
        return f"{path}: ERROR parsing as CSV ({e})"

    header = rows[0] if rows else []
    row_count = max(0, len(rows) - 1)
    sample = rows[1] if len(rows) > 1 else []
    return f"{path}: {row_count} data rows, columns={header}, sample_row={sample}"


def _guess_labels(root: Path, csv_files: list[Path]) -> str:
    # Heuristic 1: a directory whose immediate children are directories
    # containing csv files -- treat child-dir name as the label.
    for candidate_parent in sorted({p.parent.parent for p in csv_files if p.parent != root}):
        try:
            subdirs = [d for d in candidate_parent.iterdir() if d.is_dir() and d.name not in SKIP_DIR_NAMES]
        except OSError:
            continue
        if len(subdirs) >= 2:
            counts = {}
            for d in subdirs:
                n = len(list(d.glob("*.csv")))
                if n > 0:
                    counts[d.name] = n
            if len(counts) >= 2:
                return (
                    f"Inferred from directory structure under {candidate_parent}: "
                    f"one subdirectory per class, counts = {counts}"
                )

    # Heuristic 2: a 'label'/'class'/'gesture' column inside the CSVs themselves.
    label_col_names = {"label", "class", "gesture", "gesture_label", "y"}
    for p in csv_files[:MAX_CSV_FILES_SUMMARIZED]:
        try:
            with open(p, newline="", errors="replace") as f:
                lines = [l for l in f if not l.startswith("#")]
            if not lines:
                continue
            header = next(csv.reader(lines[:1]))
            matches = [h for h in header if h.strip().lower() in label_col_names]
            if matches:
                return f"Found a likely label column {matches} in {p} (spot check only -- not tallied across all files)"
        except Exception:
            continue

    return (
        "Could not automatically infer a label scheme from directory "
        "structure or an obvious label column -- rely on the README's own "
        "description of how labels are derived."
    )


def _all_script_files(root: Path) -> list[Path]:
    """Both plain scripts and notebooks -- either is fine (e.g. a
    notebook meant to run in Google Colab instead of locally)."""
    patterns = ("*.py", "*.ipynb")
    return [
        p for pattern in patterns for p in root.rglob(pattern)
        if not any(part in SKIP_DIR_NAMES for part in p.parts)
    ]


def _extract_notebook_text(path: Path) -> str:
    """Renders a .ipynb's cells (code + markdown, in source order) as
    plain text. Stored outputs, execution counts, and metadata are
    dropped -- those reflect whoever last ran the notebook (possibly
    stale, and a plot/image output can be a huge embedded base64 blob),
    not the logic being graded."""
    try:
        nb = json.loads(path.read_text(errors="replace"))
    except Exception as e:
        return f"[ERROR parsing notebook JSON: {e}]"

    cells = nb.get("cells")
    if not isinstance(cells, list):
        return "[ERROR: not a recognizable .ipynb -- no top-level 'cells' list]"

    parts = []
    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "?")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        parts.append(f"# --- cell {i} ({cell_type}) ---\n{source}")
    return "\n\n".join(parts) if parts else "(notebook has no cells)"


def _read_script_text(p: Path) -> str:
    if p.suffix.lower() == ".ipynb":
        return _extract_notebook_text(p)
    try:
        return p.read_text(errors="replace")
    except Exception as e:
        return f"[ERROR reading file: {e}]"


def _collect_dependency_manifests(root: Path) -> list[ScriptSource]:
    """Every requirements*.txt found, in full -- these are small, and
    without them a grader has no way to cross-check a script's imports
    against what you've actually declared installed."""
    manifests = [
        p for p in root.rglob("requirements*.txt")
        if not any(part in SKIP_DIR_NAMES for part in p.parts)
    ]
    manifests.sort(key=lambda p: (len(p.parts), p.name))

    included: list[ScriptSource] = []
    budget = MAX_DEPENDENCY_MANIFEST_CHARS
    for p in manifests:
        if budget <= 0:
            break
        try:
            text = p.read_text(errors="replace")
        except Exception as e:
            text = f"[ERROR reading file: {e}]"
        truncated = len(text) > budget
        text = text[:budget]
        included.append(ScriptSource(rel_path=str(p.relative_to(root)), source_text=text, truncated=truncated))
        budget -= len(text)

    return included


def _collect_script_sources(root: Path) -> tuple[list[ScriptSource], int]:
    """Gather every .py/.ipynb file's source, unfiltered by role -- the
    grader (an LLM) decides which file(s) are the augmentation/demo
    scripts itself. Prioritizes files matching a keyword heuristic first
    (so if the budget runs out on a submission with many scripts, the
    most-likely-relevant ones survive), then the rest by path. Returns
    (included_sources, omitted_count)."""
    script_files = _all_script_files(root)

    def relevance_key(p: Path):
        name_and_head = p.name.lower()
        try:
            name_and_head += _read_script_text(p)[:1000].lower()
        except Exception:
            pass
        is_relevant = any(k in name_and_head for k in AUGMENTATION_KEYWORDS + DEMO_KEYWORDS)
        return (0 if is_relevant else 1, len(p.parts), p.name)

    script_files.sort(key=relevance_key)

    included: list[ScriptSource] = []
    omitted = 0
    budget = MAX_TOTAL_SOURCE_CHARS
    for p in script_files:
        if budget <= 0:
            omitted += 1
            continue
        text = _read_script_text(p)
        truncated = False
        if len(text) > MAX_SINGLE_SOURCE_CHARS:
            text = text[:MAX_SINGLE_SOURCE_CHARS] + "\n...[truncated]..."
            truncated = True
        if len(text) > budget:
            text = text[:budget] + "\n...[truncated, total source budget reached]..."
            truncated = True
        included.append(ScriptSource(rel_path=str(p.relative_to(root)), source_text=text, truncated=truncated))
        budget -= len(text)

    return included, omitted


def analyze(root: Path) -> Analysis:
    root = root.resolve()
    all_files = list(_iter_files(root))
    csv_files = [p for p in all_files if p.suffix.lower() == ".csv"]

    readme_path = _find_readme(root)
    readme_text = None
    if readme_path is not None:
        readme_text = readme_path.read_text(errors="replace")
        if len(readme_text) > MAX_README_CHARS:
            readme_text = readme_text[:MAX_README_CHARS] + "\n...[README truncated]..."

    csv_summaries = [_summarize_csv(p) for p in csv_files[:MAX_CSV_FILES_SUMMARIZED]]
    if len(csv_files) > MAX_CSV_FILES_SUMMARIZED:
        csv_summaries.append(f"... and {len(csv_files) - MAX_CSV_FILES_SUMMARIZED} more CSV files not shown")

    label_guess = _guess_labels(root, csv_files)
    file_list = [str(p.relative_to(root)) for p in all_files[:MAX_FILES_LISTED]]
    sources, omitted = _collect_script_sources(root)
    dependency_manifests = _collect_dependency_manifests(root)

    return Analysis(
        file_list=file_list,
        total_file_count=len(all_files),
        readme_text=readme_text,
        readme_path=str(readme_path.relative_to(root)) if readme_path else None,
        csv_summaries=csv_summaries,
        label_guess=label_guess,
        script_sources=sources,
        script_sources_omitted_count=omitted,
        dependency_manifests=dependency_manifests,
    )


def render_analysis_text(a: Analysis, *, provenance: str | None = None) -> str:
    file_list_str = "\n".join(a.file_list)
    if a.total_file_count > len(a.file_list):
        file_list_str += f"\n... and {a.total_file_count - len(a.file_list)} more files not shown"

    readme_block = a.readme_text if a.readme_text is not None else "*** NO README FOUND ANYWHERE IN THE SUBMISSION ***"

    provenance_block = f"\n{provenance}\n" if provenance else ""

    if a.script_sources:
        parts = []
        for s in a.script_sources:
            trunc_note = " [TRUNCATED]" if s.truncated else ""
            parts.append(f"--- {s.rel_path}{trunc_note} ---\n{s.source_text}")
        scripts_block = "\n\n".join(parts)
        if a.script_sources_omitted_count:
            scripts_block += f"\n\n... and {a.script_sources_omitted_count} more .py/.ipynb file(s) omitted entirely (total source budget reached)"
    else:
        scripts_block = "(no .py or .ipynb files found anywhere in the submission)"

    if a.dependency_manifests:
        parts = []
        for s in a.dependency_manifests:
            trunc_note = " [TRUNCATED]" if s.truncated else ""
            parts.append(f"--- {s.rel_path}{trunc_note} ---\n{s.source_text}")
        manifests_block = "\n\n".join(parts)
    else:
        manifests_block = "(no requirements*.txt found anywhere in the submission)"

    body = f"""\
--- File listing ({a.total_file_count} files total) ---
{file_list_str}

--- README ({a.readme_path or "not found"}) ---
{readme_block}

--- CSV file summaries ({len(a.csv_summaries)} shown) ---
{chr(10).join(a.csv_summaries) if a.csv_summaries else "(no CSV files found)"}

--- Label scheme guess (heuristic, may be wrong -- cross-check against README) ---
{a.label_guess}

--- Dependency manifests (requirements*.txt, in full) ---
{manifests_block}

--- Python source (NOT executed -- static review only; see the grading prompt) ---
--- .ipynb notebooks are rendered as their code+markdown cells only -- stored outputs/metadata are stripped ---
{scripts_block}
"""

    # Hard backstop: the per-section budgets above are tuned to stay well
    # under MAX_BUNDLE_CHARS in ordinary cases, but truncate the tail here
    # regardless (rather than trusting that arithmetic) so a pathological
    # submission (e.g. unusually many/long file paths) can't produce an
    # unbounded prompt. Earlier sections (file listing, README, CSV
    # summaries) are kept intact where possible since they're smaller and
    # more information-dense; the script-source section, which is both the
    # largest contributor and already independently capped, absorbs the cut.
    if len(body) > MAX_BUNDLE_CHARS:
        body = (
            body[:MAX_BUNDLE_CHARS]
            + f"\n\n...[BUNDLE TRUNCATED: exceeded the {MAX_BUNDLE_CHARS}-character hard limit -- "
            "some script source and/or CSV summaries below this point were cut off]...\n"
        )

    return (
        f"===== SUBMISSION ANALYSIS (produced by build_bundle.py, no AI involved) ====={provenance_block}\n\n"
        f"{body}\n"
        "===== END SUBMISSION ANALYSIS =====\n"
    )


if __name__ == "__main__":
    # CLI: `python3 build_bundle.py <submission_dir> [--out bundle.txt]`
    # Nothing in the submission is ever executed by this CLI, only read.
    import argparse
    import getpass
    import socket
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Output bundle path (default: <submission_dir>_bundle.txt)")
    args = ap.parse_args()

    submission_dir = args.submission_dir.resolve()
    out_path = args.out or submission_dir.parent / f"{submission_dir.name}_bundle.txt"

    try:
        generated_by = f"{getpass.getuser()}@{socket.gethostname()}"
    except Exception:
        generated_by = "unknown"
    provenance = (
        f"Bundle generated: {datetime.now(timezone.utc).isoformat()} by {generated_by}\n"
        f"Submission directory name at generation time: {submission_dir.name}\n"
        "Nothing in the submission was executed to produce this bundle; "
        "it's a text snapshot of the README, file listing, and script source (.py/.ipynb)."
    )

    result = analyze(submission_dir)
    out_path.write_text(render_analysis_text(result, provenance=provenance))
    print(f"Wrote bundle to {out_path}")
