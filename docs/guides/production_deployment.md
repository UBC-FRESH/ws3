# Production Deployment Guide

**Version**: 1.1.0a1  
**Date**: 2026-07-26  
**Status**: In Progress  

## Overview

This guide documents the production deployment process for ws3, including:
- Release packaging and distribution
- Continuous Integration/Continuous Deployment (CI/CD)
- Version management and semantic versioning
- Community guidelines and support channels

## Release Packaging

### Package Structure

ws3 follows Python packaging conventions:

```
ws3/
├── pyproject.toml          # Package metadata and dependencies
├── setup.py                # Legacy setup (optional)
├── src/
│   └── ws3/
│       ├── __init__.py     # Version info
│       ├── forest.py       # Core modeling
│       ├── opt.py          # Optimization
│       ├── perf.py         # Performance optimization
│       └── integration.py  # External integrations
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
└── examples/               # Example notebooks and scripts
```

### Version Management

**Current Version**: 1.1.0a1

Version follows Semantic Versioning (SemVer):
- **MAJOR**: Incompatible API changes (1.0.0 → 2.0.0)
- **MINOR**: Backward-compatible features (2.0.0 → 2.1.0)
- **PATCH**: Backward-compatible bug fixes (2.0.0 → 2.0.1)

**Version Location**: `ws3/__init__.py`

```python
__version__ = "2.0.0"
__author__ = "UBC-FRESH Team"
__license__ = "MIT"
```

### Building Packages

**Source Distribution (sdist)**:
```bash
python -m build --sdist
```

**Wheel Distribution**:
```bash
python -m build --wheel
```

**Both**:
```bash
python -m build
```

**Output**: `dist/` directory contains:
- `ws3-2.0.0.tar.gz` (source distribution)
- `ws3-2.0.0-py3-none-any.whl` (wheel)

### Testing Before Release

**Run all tests**:
```bash
pytest tests/ -v
```

**Check documentation build**:
```bash
sphinx-build -b html docs/source docs/build/html
```

**Validate package**:
```bash
twine check dist/*
```

## CI/CD Pipeline

### GitHub Actions Workflow

**.github/workflows/ci.yml**:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest tests/ -v
    
    - name: Check documentation
      run: sphinx-build -b html docs/source docs/build/html
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: test-results-${{ matrix.python-version }}
        path: tests/results/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.12"
    
    - name: Install build dependencies
      run: pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: dist
        path: dist/
  
  publish:
    needs: build
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Download artifacts
      uses: actions/download-artifact@v4
      with:
        name: dist
        path: dist/
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### Release Process

**1. Create release branch**:
```bash
git checkout -b release/v2.0.0 develop
```

**2. Update version**:
```bash
# Edit ws3/__init__.py
# Update CHANGELOG.md
# Update ROADMAP.md
```

**3. Run tests**:
```bash
pytest tests/ -v
sphinx-build -b html docs/source docs/build/html
```

**4. Build package**:
```bash
python -m build
twine check dist/*
```

**5. Test installation**:
```bash
pip install dist/ws3-2.0.0-py3-none-any.whl
python -c "import ws3; print(ws3.__version__)"
```

**6. Create git tag**:
```bash
git tag -a v2.0.0 -m "Release v2.0.0"
git push origin v2.0.0
```

**7. Publish to PyPI** (automated via CI/CD):
```bash
# Automatic on tag push
```

**8. Create GitHub Release**:
- Go to GitHub → Releases → Create new release
- Select tag: `v2.0.0`
- Add release notes from CHANGELOG.md
- Attach distribution files (optional)

## Version History

### v2.0.0 (2026-07-26)

**Major Features**:
- Phase 5 interactive notebooks (070-077)
- Performance optimization module (`ws3.perf`)
- Integration module (`ws3.integration`)
- FAQ section and migration guide
- Multi-objective optimization examples
- Spatial constraints and adjacency modeling
- Parallel optimization and benchmarking

**Breaking Changes**:
- None (backward compatible with 1.x)

**New Modules**:
- `ws3.perf`: Performance optimization utilities
- `ws3.integration`: External tool integrations

**Enhanced Documentation**:
- FAQ section (20 common questions)
- Woodstock migration guide
- 16 interactive notebooks
- 14 how-to guides

### v1.0.0 (2024-11-24)

**Initial Release**:
- Core forest modeling (`ws3.forest`)
- Optimization module (`ws3.opt`)
- Documentation system
- Basic examples

## Community Guidelines

### Contribution Process

**1. Fork and Clone**:
```bash
git clone https://github.com/your-username/ws3.git
cd ws3
```

**2. Create Feature Branch**:
```bash
git checkout -b feature/your-feature-name
```

**3. Make Changes**:
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation
- Run linters and type checkers

**4. Test**:
```bash
pytest tests/ -v
sphinx-build -b html docs/source docs/build/html
```

**5. Commit**:
```bash
git add .
git commit -m "feat: add your feature description"
```

**6. Push and Pull Request**:
```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

### Code Style

**Python**:
- PEP 8 compliant
- Type hints required
- Docstrings for all public functions
- Maximum line length: 100 characters

**Commits**:
- Conventional Commits format
- Clear, descriptive commit messages
- One logical change per commit

**Documentation**:
- RST format for docs
- Examples for new features
- Update CHANGELOG.md

### Issue Reporting

**Bug Reports**:
- Use GitHub Issues → Bug Report template
- Include: ws3 version, Python version, solver
- Provide minimal reproducible example
- Include full error traceback

**Feature Requests**:
- Use GitHub Issues → Feature Request template
- Describe use case and expected behavior
- Link to related issues if applicable

### Support Channels

**Primary**:
- GitHub Issues: https://github.com/UBC-FRESH/ws3/issues
- GitHub Discussions: https://github.com/UBC-FRESH/ws3/discussions

**Secondary**:
- Email: ws3@forestry.ubc.ca
- Slack/Discord: (to be established)

**Documentation**:
- ReadTheDocs: https://ws3.readthedocs.io
- Examples: `examples/` directory
- Notebooks: `examples/*.ipynb`

## Deployment Checklist

### Pre-Release

- [ ] All tests passing
- [ ] Documentation builds without errors
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] ROADMAP.md updated
- [ ] LICENSE file present
- [ ] README.md current
- [ ] pyproject.toml dependencies correct
- [ ] No deprecated APIs
- [ ] Performance benchmarks run
- [ ] Security audit completed

### Release

- [ ] Tag created (`vX.Y.Z`)
- [ ] GitHub Release created
- [ ] Release notes written
- [ ] Distribution files uploaded to PyPI
- [ ] Conda package published (if applicable)
- [ ] Documentation deployed to ReadTheDocs

### Post-Release

- [ ] Announce on community channels
- [ ] Update website/brochure
- [ ] Monitor issue tracker for post-release bugs
- [ ] Plan next release
- [ ] Archive release branch

## Troubleshooting

### Common Issues

**Issue**: Tests fail on CI but pass locally
**Solution**: Check Python version matrix, ensure all dependencies installed

**Issue**: Documentation build fails
**Solution**: Check RST syntax, verify all cross-references exist

**Issue**: Package installation fails
**Solution**: Check Python version compatibility, verify dependencies

**Issue**: Type checking errors
**Solution**: Run `mypy ws3/` and fix type annotations

### Getting Help

1. Check existing issues and discussions
2. Review documentation and examples
3. Search error messages online
4. Create new issue with detailed information

## Future Roadmap

### v2.1.0 (Planned)
- Enhanced carbon accounting
- Dynamic planning capabilities
- Climate scenario integration
- Improved spatial modeling

### v3.0.0 (Planned)
- Major API redesign
- Plugin architecture
- Distributed computing support
- Cloud deployment options

---

**Last Updated**: 2026-07-26  
**Maintainer**: UBC-FRESH Team  
**Contact**: ws3@forestry.ubc.ca