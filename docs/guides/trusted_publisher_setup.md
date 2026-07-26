# GitHub Trusted Publisher Setup for ws3

## Overview

This guide sets up GitHub Actions to publish ws3 to TestPyPI and PyPI using GitHub's Trusted Publisher pattern (OIDC tokens) instead of API tokens.

**Reference:** This follows the same pattern as `freshforge`.

---

## Step 1: Configure GitHub Repository Settings

### 1.1 Create Environments

Go to **Settings → Environments** in the ws3 GitHub repository:

#### Create `pypi` Environment:
1. Click **New environment**
2. Name: `pypi`
3. Environment URL: `https://pypi.org/project/ws3/`
4. Click **Configure environment**
5. Under **Required reviewers**, leave empty (no reviewers needed)
6. Under **Deployment branches**, select **All branches** (or restrict to main only)
7. Click **Save environment**

#### Create `testpypi` Environment (optional):
1. Click **New environment**
2. Name: `testpypi`
3. Environment URL: `https://test.pypi.org/project/ws3/`
4. Click **Configure environment**
5. Click **Save environment**

### 1.2 Configure Trusted Publishers

#### For PyPI:
1. Go to **https://pypi.org/manage/account/publishing/**
2. Click **Add a new project publisher**
3. Fill in:
   - **Owner or organization**: `gep-ubc`
   - **Repository name**: `ws3`
   - **Environment name**: `pypi` (must match the GitHub environment name)
   - **Workflow filename prefix**: `ci` (matches `ci.yml`)
4. Click **Add publisher**

#### For TestPyPI (optional):
1. Go to **https://test.pypi.org/manage/account/publishing/**
2. Click **Add a new project publisher**
3. Fill in:
   - **Owner or organization**: `gep-ubc`
   - **Repository name**: `ws3`
   - **Environment name**: `testpypi`
   - **Workflow filename prefix**: `ci`
4. Click **Add publisher**

---

## Step 2: Update CI Workflow

The CI workflow (`ci.yml`) has been updated with the trusted publisher pattern:

```yaml
publish-pypi:
  name: Publish to PyPI
  needs: build
  if: startsWith(github.ref, 'refs/tags/v')
  runs-on: ubuntu-latest
  environment:
    name: pypi
    url: https://pypi.org/project/ws3/
  permissions:
    contents: read
    id-token: write  # Required for OIDC

  steps:
    - name: Download distribution files
      uses: actions/download-artifact@v4
      with:
        name: dist
        path: dist/

    - name: Publish package distributions to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

**Key changes:**
- Removed `password: ${{ secrets.PYPI_API_TOKEN }}`
- Added `environment: pypi`
- Added `permissions: id-token: write`
- Added `url` to environment for tracking

---

## Step 3: Verify the Setup

### 3.1 Test with a Dry Run

Create a test tag (don't push to main):

```bash
cd /home/gep/projects/ws3
git tag v1.1.0b1-test
git push origin v1.1.0b1-test
```

This will trigger the CI workflow. Monitor the **Actions** tab to see:
1. Build job completes
2. Publish to PyPI job runs
3. OIDC token is exchanged for PyPI credentials
4. Package is published

### 3.2 Verify on PyPI

After the workflow completes:
1. Visit https://pypi.org/project/ws3/
2. Check that the new version appears

---

## Step 4: Cleanup Test Release

If the test was successful:

```bash
git tag -d v1.1.0b1-test
git push origin :refs/tags/v1.1.0b1-test
```

---

## Step 5: Publish to Production

Once you're ready to publish to production PyPI:

```bash
# Bump version in pyproject.toml (e.g., 1.1.0a1 → 1.1.0)
# Update CHANGELOG.md with release notes
git add .
git commit -m "Release v1.1.0"
git tag v1.1.0
git push origin v1.1.0
```

The CI workflow will automatically:
1. Build the package
2. Run tests
3. Publish to PyPI using trusted publisher

---

## Troubleshooting

### Error: "No such file or directory: 'pypa/gh-action-pypi-publish'"

**Cause:** The action is not available in the runner.

**Solution:** Ensure you're using `pypa/gh-action-pypi-publish@release/v1` (not an older version).

### Error: "Missing required parameter: 'password'"

**Cause:** The workflow is still using API token authentication.

**Solution:** Remove the `password` parameter from the publish step. The trusted publisher pattern uses OIDC instead.

### Error: "Repository not found"

**Cause:** The trusted publisher is not configured correctly on PyPI.

**Solution:**
1. Verify the repository name is `ws3` (not `ws3-package` or similar)
2. Verify the owner is `gep-ubc`
3. Verify the environment name matches exactly (`pypi`)

### Error: "Invalid environment name"

**Cause:** The environment name in GitHub doesn't match the one configured on PyPI.

**Solution:** Ensure both use the same name (e.g., `pypi`).

---

## Security Benefits

**Trusted Publisher (OIDC) vs API Token:**

| Aspect | API Token | Trusted Publisher (OIDC) |
|--------|-----------|--------------------------|
| Storage | Stored as secret | No secret needed |
| Rotation | Manual rotation required | Automatic |
| Scope | Can be overly broad | Tied to specific workflow |
| Compromise | Token can be abused | Only valid for workflow run |
| Setup | More complex | Simpler |

---

## References

- [GitHub Actions OIDC with PyPI](https://docs.github.com/en/packages/working-with-a-github-packages-registry/ publishing-and-installing-a-package-with-github-actions)
- [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)
- [freshforge release workflow](https://github.com/gep-ubc/freshforge/blob/main/.github/workflows/release.yml)

---

## Checklist

- [ ] Create `pypi` environment in GitHub repository settings
- [ ] Add trusted publisher on PyPI (https://pypi.org/manage/account/publishing/)
- [ ] Test with a beta tag (e.g., `v1.1.0b1-test`)
- [ ] Verify package appears on PyPI
- [ ] Clean up test tag
- [ ] Document in CHANGELOG.md
- [ ] Mark as complete in planning/phase5_status_audit.md