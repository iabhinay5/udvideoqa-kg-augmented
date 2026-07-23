#!/usr/bin/env python3
"""
compile_report.py — Compile the LaTeX report to PDF.

Three methods (tries in order):
  1. Local pdflatex (if installed: MiKTeX or TeX Live)
  2. Docker with texlive (if Docker is installed)
  3. Online: prints instructions for Overleaf upload

Usage:
  python scripts/compile_report.py
"""
import subprocess
import shutil
from pathlib import Path

REPORT_TEX = Path("report/report.tex")
REPORT_DIR = Path("report")


def try_local_pdflatex():
    """Try compiling with local pdflatex."""
    exe = shutil.which("pdflatex")
    if not exe:
        return False
    print(f"Found pdflatex at: {exe}")
    for _ in range(2):  # two passes for cross-references
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            cwd=str(REPORT_DIR), capture_output=True, text=True
        )
    if (REPORT_DIR / "report.pdf").exists():
        print("SUCCESS: report/report.pdf generated.")
        return True
    print("pdflatex run but PDF not found. Check report/report.log")
    return False


def try_docker():
    """Try compiling with Docker texlive image."""
    exe = shutil.which("docker")
    if not exe:
        return False
    print("Docker found. Trying texlive container...")
    result = subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{Path('report').resolve().as_posix()}:/doc",
        "texlive/texlive:latest",
        "pdflatex", "-interaction=nonstopmode",
        "/doc/report.tex",
    ], capture_output=True, text=True, timeout=120)
    if (REPORT_DIR / "report.pdf").exists():
        print("SUCCESS: report/report.pdf generated via Docker.")
        return True
    return False


def print_overleaf_instructions():
    """Tell user how to upload to Overleaf."""
    print("""
=====================================
LaTeX not found locally or via Docker.
=====================================

Option 1 — Overleaf (easiest):
  1. Go to https://overleaf.com and sign in (free account)
  2. Click 'New Project' -> 'Upload Project'
  3. Zip the 'report/' folder and upload it
  4. Also upload all 'figures/' PDF files to the 'figures/' folder in Overleaf
  5. Click 'Recompile'

Option 2 — Install MiKTeX (Windows):
  1. Download from https://miktex.org/download
  2. Install MiKTeX (includes pdflatex)
  3. Run: python scripts/compile_report.py

Option 3 — Use a remote Linux server (LaTeX usually pre-installed):
  scp -r report/ figures/ user@your-server:/path/to/capstone/
  ssh user@your-server
  cd /path/to/capstone/report
  pdflatex report.tex && pdflatex report.tex
  scp report.pdf /your/local/machine/

The .tex file is fully self-contained (bibliography embedded).
No .bib file needed.
""")


def main():
    print(f"Compiling {REPORT_TEX}...")
    if try_local_pdflatex():
        return
    if try_docker():
        return
    print_overleaf_instructions()


if __name__ == "__main__":
    main()
