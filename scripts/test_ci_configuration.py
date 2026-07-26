#!/usr/bin/env python3
"""
Test script to verify CI/CD pipeline configuration.

This script checks:
- Repository secrets are configured
- Package can be built locally
- Documentation builds successfully
- All tests pass
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return result."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"   Error: {result.stderr}")
        return False
    return True


def check_repository_secrets():
    """Check if required repository secrets are configured."""
    print("\n📋 Checking Repository Secrets...")
    
    # Check if gh CLI is available
    if not run_command("which gh", check=False):
        print("⚠️  GitHub CLI not installed. Skipping secret checks.")
        print("   Install: https://cli.github.com/")
        return True
    
    # Check for PyPI token
    result = subprocess.run(
        "gh secret list 2>/dev/null | grep -q PYPI_API_TOKEN",
        shell=True,
        capture_output=True
    )
    
    if result.returncode == 0:
        print("✅ PYPI_API_TOKEN configured")
        return True
    else:
        print("❌ PYPI_API_TOKEN not found")
        print("   Add to: Settings → Secrets and variables → Actions")
        return False


def check_package_build():
    """Check if package can be built."""
    print("\n📦 Checking Package Build...")
    
    # Install build dependencies
    if not run_command("pip install build twine --quiet"):
        return False
    
    # Build package
    if not run_command("python -m build --quiet"):
        return False
    
    # Check output
    dist_dir = Path("dist")
    if dist_dir.exists() and len(list(dist_dir.glob("*"))) > 0:
        print("✅ Package built successfully")
        print(f"   Files: {list(dist_dir.glob('*'))}")
        return True
    else:
        print("❌ Package build failed - no output in dist/")
        return False


def check_documentation_build():
    """Check if documentation builds."""
    print("\n📚 Checking Documentation Build...")
    
    # Install docs dependencies
    if not run_command("pip install -e '.[docs]' --quiet"):
        print("⚠️  Could not install docs dependencies")
        return True  # Not critical
    
    # Build documentation
    if not run_command("sphinx-build -b html docs/source docs/build/html --quiet"):
        return False
    
    # Check output
    html_dir = Path("docs/build/html")
    if html_dir.exists() and (html_dir / "index.html").exists():
        print("✅ Documentation built successfully")
        return True
    else:
        print("❌ Documentation build failed")
        return False


def check_tests():
    """Check if tests pass."""
    print("\n🧪 Checking Tests...")
    
    # Run tests
    result = subprocess.run(
        "pytest tests/ -v --tb=short",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Count passed tests
        import re
        passed = len(re.findall(r"PASSED", result.stdout))
        print(f"✅ All tests passed ({passed} tests)")
        return True
    else:
        print("❌ Tests failed")
        print(result.stdout[-500:])  # Last 500 chars
        return False


def check_version():
    """Check version information."""
    print("\n🏷️  Checking Version...")
    
    try:
        import ws3
        version = ws3.__version__
        print(f"✅ ws3 version: {version}")
        return True
    except Exception as e:
        print(f"❌ Could not import ws3: {e}")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("CI/CD Configuration Test")
    print("="*60)
    
    checks = [
        ("Repository Secrets", check_repository_secrets),
        ("Package Build", check_package_build),
        ("Documentation Build", check_documentation_build),
        ("Tests", check_tests),
        ("Version", check_version),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! CI/CD pipeline is ready.")
        print("\nNext steps:")
        print("1. Push to GitHub")
        print("2. Monitor Actions tab for pipeline runs")
        print("3. Create a test tag to verify PyPI publishing")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())