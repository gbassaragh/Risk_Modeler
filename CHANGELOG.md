# Changelog

All notable changes to Risk_Modeler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-29

### Added
- Initial release of Risk_Modeler
- Monte Carlo simulation engine with LHS, Sobol, and standard MC sampling
- Support for multiple probability distributions (Triangular, PERT, Normal, Log-Normal, Uniform, Discrete)
- WBS-based cost modeling for utility T&D projects
- Risk driver methodology with occurrence and impact modeling
- Iman-Conover correlation transformation
- Comprehensive I/O support (Excel, CSV, JSON)
- CLI interface with multiple commands (`run`, `template`, `validate`, `report`, `verify`)
- Docker containerization support with multi-stage builds
- CI/CD pipeline with GitHub Actions
- Performance optimization with Numba JIT compilation
- Comprehensive test suite with >80% coverage
- Two-layer uncertainty modeling for epistemic/aleatory separation
- Advanced distribution fitting and calibration capabilities
- Sensitivity analysis with tornado diagrams and Spearman correlation
- Audit trail and determinism verification
- Statistical validation with convergence diagnostics
- Professional reporting with P10/P50/P80/P90 percentiles

### Technical Features
- Pydantic v2 data models with comprehensive validation
- NumPy/SciPy optimized numerical computing
- Latin Hypercube Sampling (LHS) for superior convergence
- Variance reduction techniques (antithetic variates)
- Memory-efficient large-scale simulations (50K+ iterations)
- Cross-platform compatibility (Windows, macOS, Linux)

### Documentation
- Comprehensive user guide with examples
- Data dictionary for all input formats
- API documentation for programmatic usage
- Docker deployment guides
- Contributing guidelines and code of conduct

### Security
- Input validation for all user data
- Secure random number generation
- No hardcoded credentials or secrets
- Dependency vulnerability scanning
- Container security best practices