# CI/CD Configuration Guide

**Version**: 2.0.0  
**Date**: 2026-07-26  
**Status**: Configuration Required  

## Overview

This guide documents how to configure the CI/CD pipeline with repository secrets and test the automated publishing process for ws3.

## Prerequisites

1. GitHub repository with admin access
2. PyPI account with upload permissions
3. Git installed locally
4. Python 3.9+ installed

## Repository Secrets Configuration

### 1. PyPI API Token

**Purpose**: Authenticate with PyPI for package publishing

**Steps**:

1. **Create PyPI API Token**:
   - Go to https://pypi.org/account/settings/#api-tokens
   - Click "Add API token"
   - Name: `ws3-ci-token`
   - Scope: `Entire account`
   - Click "Create token"
   - **Copy the token immediately** (only shown once)

2. **Add to GitHub Repository Secrets**:
   - Go to your GitHub repository
   - Navigate to: Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: `<paste-your-pypi-token-here>`
   - Click "Add secret"

3. **Verify Secret**:
   ```bash
   # Check if secret is configured (requires GitHub CLI)
   gh secret list
   ```

### 2. Optional: ReadTheDocs Token

**Purpose**: Auto-deploy documentation to ReadTheDocs

**Steps**:

1. **Get ReadTheDocs Token**:
   - Go to https://readthedocs.org/dashboard/ws3/settings/
   - Navigate to "Builds" → "Webhooks"
   - Copy the webhook URL or token

2. **Add to GitHub Repository Secrets**:
   - Settings → Secrets and variables → Actions
   - Name: `READTHEDOCS_TOKEN`
   - Value: `<your-readthedocs-token>`
   - Click "Add secret"

### 3. Optional: Slack/Discord Webhook

**Purpose**: Notify team on pipeline completion

**Steps**:

1. **Create Webhook**:
   - Slack: Settings → Integrations → Webhooks
   - Discord: Server Settings → Integrations → Webhooks

2. **Add to GitHub Repository Secrets**:
   - Name: `SLACK_WEBHOOK_URL` or `DISCORD_WEBHOOK_URL`
   - Value: `<webhook-url>`
   - Click "Add secret"

## Testing the Pipeline

### 1. Test Local Build

```bash
# Install build dependencies
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Verify package contents
ls -lh dist/
```

**Expected Output**:
```
creating 'dist/ws3-2.0.0.tar.gz' and adding 'build/lib/ws3' to it
creating 'dist/ws3-2.0.0-py3-none-any.whl' and adding 'build/lib/ws3' to it
```

### 2. Test Installation

```bash
# Install from wheel
pip install dist/ws3-2.0.0-py3-none-any.whl

# Verify installation
python -c "import ws3; print(ws3.__version__)"

# Expected: 2.0.0
```

### 3. Test Documentation Build

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
sphinx-build -b html docs/source docs/build/html

# Check for errors
echo $?  # Should be 0
```

### 4. Test on GitHub Actions

**Push a test commit**:
```bash
git add .
git commit -m "test: verify CI/CD pipeline configuration"
git push origin feature/ws3-phase5
```

**Monitor Pipeline**:
- Go to GitHub repository → Actions tab
- Watch the pipeline run
- Check each job: lint, test, docs, build

**Expected Jobs**:
1. ✅ Lint and Type Check
2. ✅ Test (Python 3.9, 3.10, 3.11, 3.12)
3. ✅ Build Documentation
4. ✅ Build Package

### 5. Test Release Process

**Create a test release branch**:
```bash
git checkout -b release/v2.0.0-test
git tag -a v2.0.0-test -m "Test release"
git push origin release/v2.0.0-test
git push origin v2.0.0-test
```

**Monitor Pipeline**:
- The `publish-pypi` job should trigger
- Check if package is published to PyPI
- Verify at https://pypi.org/project/ws3/

**Cleanup** (if needed):
```bash
# Delete test tag and branch
git push origin --delete v2.0.0-test
git push origin --delete release/v2.0.0-test
git tag -d v2.0.0-test
```

## Troubleshooting

### Issue: PyPI Upload Fails with "Invalid Authentication"

**Cause**: Incorrect or expired PyPI token

**Solution**:
1. Verify token in repository secrets
2. Create new token on PyPI if needed
3. Update secret in GitHub repository

### Issue: Build Job Fails with "Module Not Found"

**Cause**: Missing dependencies in pyproject.toml

**Solution**:
1. Check pyproject.toml dependencies
2. Add missing dependencies
3. Test locally before pushing

### Issue: Test Job Fails on Specific Python Version

**Cause**: Incompatibility with Python version

**Solution**:
1. Check test output for specific error
2. Update code for compatibility
3. Or exclude that Python version in workflow

### Issue: Documentation Build Fails

**Cause**: RST syntax errors or missing cross-references

**Solution**:
1. Check sphinx-build output for errors
2. Fix RST syntax
3. Add missing references or files

### Issue: Pipeline Stuck or Timeout

**Cause**: Long-running tests or builds

**Solution**:
1. Increase timeout in workflow
2. Optimize test suite
3. Split into multiple jobs

## Security Best Practices

### 1. Token Rotation

**Frequency**: Every 90 days or when team members leave

**Process**:
1. Create new token on PyPI
2. Update GitHub secret
3. Delete old token on PyPI
4. Test pipeline with new token

### 2. Secret Scanning

**Enable GitHub Secret Scanning**:
- Settings → Code security and analysis
- Enable "Secret scanning"
- Enable "Push protection"

**Pre-commit Hook** (optional):
```bash
# Install detect-secrets
pip install detect-secrets

# Add to .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
```

### 3. Least Privilege

**PyPI Token Scope**:
- Use "Entire account" only if needed
- Consider "Specific project" for limited scope

**GitHub Actions Permissions**:
- Settings → Actions → General
- Set minimum permissions for workflows

## Monitoring and Maintenance

### 1. Pipeline Status

**Check Recent Runs**:
- GitHub → Actions → All Workflows
- Filter by branch or status

**Set Up Notifications**:
- GitHub → Settings → Notifications
- Configure email or Slack notifications

### 2. Dependency Updates

**Automated Updates**:
- Enable Dependabot: Settings → Code security and analysis
- Enable "Dependabot alerts"
- Enable "Dependabot PRs"

**Manual Updates**:
```bash
# Update dependencies
pip install --upgrade ws3

# Check for updates
pip list --outdated
```

### 3. Version Tracking

**Monitor PyPI**:
- https://pypi.org/project/ws3/#history
- Check download statistics
- Monitor for abuse or unusual activity

**GitHub Releases**:
- Settings → General → GitHub Pages (if enabled)
- Monitor release activity
- Track issue reports per version

## Next Steps

1. **Configure Repository Secrets**:
   - Add PYPI_API_TOKEN
   - (Optional) Add READTHEDOCS_TOKEN
   - (Optional) Add SLACK_WEBHOOK_URL

2. **Test Pipeline**:
   - Push test commit
   - Verify all jobs pass
   - Check artifacts

3. **Create Test Release**:
   - Create test tag
   - Verify PyPI publishing
   - Cleanup test artifacts

4. **Document Team Access**:
   - Share secret management process
   - Document token rotation schedule
   - Train team on troubleshooting

5. **Enable Monitoring**:
   - Set up notifications
   - Configure Dependabot
   - Schedule regular reviews

---

**Last Updated**: 2026-07-26  
**Maintainer**: UBC-FRESH Team  
**Contact**: ws3@forestry.ubc.ca