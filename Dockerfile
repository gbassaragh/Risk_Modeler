# Multi-stage build for Risk Modeling Tool
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir build

# Copy source code
COPY . .

# Build the package
RUN python -m build

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r riskuser && useradd -r -g riskuser riskuser

# Set working directory
WORKDIR /app

# Copy built package from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir *.whl && \
    rm -rf *.whl

# Create directories for templates and data
RUN mkdir -p /app/templates /app/data /app/results && \
    chown -R riskuser:riskuser /app

# Switch to non-root user
USER riskuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit (if used)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import risk_tool; print('OK')" || exit 1

# Default command
CMD ["risk-tool", "--help"]