# Contributing to Risk_Modeler

Thank you for your interest in contributing to Risk_Modeler!

## Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism

## How to Contribute

### Reporting Issues
1. Check existing issues first
2. Use issue templates when available
3. Provide minimal reproducible examples

### Pull Requests
1. Fork the repository
2. Create a feature branch from `develop`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Run tests and ensure they pass
   ```bash
   pytest risk_tool/tests/
   ```
5. Update documentation if needed
6. Commit with conventional commit messages
7. Push to your fork and submit a PR to `develop` branch

## Development Setup
```bash
# Clone repository
git clone https://github.com/gbassaragh/Risk_Modeler.git
cd Risk_Modeler

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## Coding Standards
- Follow PEP 8
- Use type hints (Python 3.11+)
- Write docstrings for all functions/classes
- Maintain test coverage above 80%
- Use Black for code formatting
- Run mypy for type checking

## Testing
- Write tests for new features
- Update tests when modifying existing features
- Run full test suite before submitting PR

## Commit Message Format
Follow conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` adding or modifying tests
- `chore:` maintenance tasks
- `ci:` CI/CD changes
- `perf:` performance improvements

## Release Process
1. Update CHANGELOG.md
2. Update version in pyproject.toml
3. Create release PR to main
4. Tag release after merge
5. GitHub Actions will handle deployment