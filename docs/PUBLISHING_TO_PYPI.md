# Publishing to PyPI - Quick Guide

## Prerequisites

1. **Create PyPI Account**
   - Go to https://pypi.org/account/register/
   - Verify your email
   - (Optional) Set up 2FA for security

2. **Install Build Tools**
   ```bash
   pip install --upgrade build twine
   ```

## Publishing Steps

### 1. Clean Previous Builds
```bash
rm -rf dist/ build/ src/*.egg-info
```

### 2. Update Version
Edit `setup.py` and bump the version number:
```python
version="1.0.0",  # Change this
```

### 3. Build the Package
```bash
python -m build
```

This creates:
- `dist/hlaprotbert-1.0.0.tar.gz` (source distribution)
- `dist/hlaprotbert-1.0.0-py3-none-any.whl` (wheel)

### 4. Check the Build
```bash
twine check dist/*
```

Should output: `Checking dist/hlaprotbert-1.0.0.tar.gz: PASSED`

### 5. Test Upload (TestPyPI - Optional but Recommended)
```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ hlaprotbert
```

### 6. Upload to PyPI
```bash
twine upload dist/*
```

You'll be prompted for:
- Username: (your PyPI username)
- Password: (your PyPI password or API token)

### 7. Verify Installation
```bash
# In a fresh environment
pip install hlaprotbert

# Test import
python -c "from hlaprotbert.models.encoders import ProtBERTEncoder; print('Success!')"
```

## Using API Tokens (Recommended)

Instead of username/password, use API tokens:

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Create `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcCJGFiY2RlZi0xMjM0LTU2NzgtOTBhYi1jZGVmZ2hpamtsbW8...
```

Then just run:
```bash
twine upload dist/*
```

## Automation Script

Save this as `scripts/publish_to_pypi.sh`:

```bash
#!/bin/bash
set -e

echo "Publishing hlaprotbert to PyPI"
echo "================================"

# Clean
echo "Cleaning previous builds..."
rm -rf dist/ build/ src/*.egg-info

# Build
echo "Building package..."
python -m build

# Check
echo "Checking package..."
twine check dist/*

# Upload
echo "Uploading to PyPI..."
read -p "Ready to upload? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    twine upload dist/*
    echo "✅ Package published successfully!"
else
    echo "❌ Upload cancelled"
    exit 1
fi
```

Make it executable:
```bash
chmod +x scripts/publish_to_pypi.sh
```

## Checklist Before Publishing

- [ ] All tests pass: `pytest`
- [ ] Version bumped in `setup.py`
- [ ] CHANGELOG.md updated
- [ ] README.md is current
- [ ] LICENSE file exists
- [ ] Git tag created: `git tag v1.0.0 && git push --tags`
- [ ] No sensitive data in package
- [ ] `.gitignore` excludes data files

## Post-Publication

1. **Verify Installation**
   ```bash
   pip install hlaprotbert
   ```

2. **Update README Badges**
   ```markdown
   [![PyPI version](https://badge.fury.io/py/hlaprotbert.svg)](https://badge.fury.io/py/hlaprotbert)
   ```

3. **Announce**
   - GitHub release notes
   - Project documentation
   - Relevant forums/communities

## Version Numbering (Semantic Versioning)

- `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking API changes (e.g., 1.0.0 → 2.0.0)
- **MINOR**: New features, backwards compatible (e.g., 0.2.0 → 0.3.0)
- **PATCH**: Bug fixes, backwards compatible (e.g., 0.3.0 → 0.3.1)

Current version: `1.0.0`  
Recommended next: `1.0.1` (patch) or `1.1.0` (minor features)

## Troubleshooting

### "Package already exists"
- You can't re-upload the same version
- Bump version number in `setup.py`
- Delete old dist files: `rm -rf dist/`

### "Invalid distribution"
- Run `twine check dist/*` to see specific errors
- Common issues:
  - Missing README.md
  - Invalid metadata in setup.py
  - Syntax errors in long_description

### "Authentication failed"
- Check username/password
- Use API token instead
- Verify account email is confirmed

## More Resources

- PyPI Guide: https://packaging.python.org/tutorials/packaging-projects/
- Twine Docs: https://twine.readthedocs.io/
- setuptools Docs: https://setuptools.pypa.io/
