#!/usr/bin/env python3
"""
Coverage runner script for the convolutional diffusion project.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_coverage(html=True, xml=True, term=True):
    """Run pytest with coverage."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=src",
        "--cov-report=term-missing" if term else "",
        "--cov-report=html:htmlcov" if html else "",
        "--cov-report=xml" if xml else "",
    ]
    
    # Remove empty strings
    cmd = [arg for arg in cmd if arg]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("\n✅ Coverage check completed successfully!")
        if html:
            print("📊 HTML report generated in htmlcov/")
        if xml:
            print("📊 XML report generated in coverage.xml")
    else:
        print("\n❌ Coverage check failed!")
        sys.exit(result.returncode)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run coverage checks")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    parser.add_argument("--no-xml", action="store_true", help="Skip XML report")
    parser.add_argument("--no-term", action="store_true", help="Skip terminal report")
    
    args = parser.parse_args()
    
    run_coverage(
        html=not args.no_html,
        xml=not args.no_xml,
        term=not args.no_term
    )


if __name__ == "__main__":
    main() 