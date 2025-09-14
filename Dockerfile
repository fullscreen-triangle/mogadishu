# Mogadishu S-Entropy Framework - Production Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM rust:1.75-slim as builder

# Set working directory
WORKDIR /usr/src/mogadishu

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build dependencies (cached layer)
RUN mkdir src/bin && echo 'fn main() {}' > src/bin/main.rs
RUN cargo build --release --features "oxygen-enhanced,quantum-transport,atp-constraints"
RUN rm src/bin/main.rs

# Build application
COPY src/bin ./src/bin
RUN cargo build --release --features "oxygen-enhanced,quantum-transport,atp-constraints"

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false -m -d /var/lib/mogadishu mogadishu

# Set working directory
WORKDIR /opt/mogadishu

# Copy built application
COPY --from=builder /usr/src/mogadishu/target/release/mogadishu-cli ./bin/
COPY --from=builder /usr/src/mogadishu/target/release/deps/libmogadishu-*.so ./lib/

# Create Python environment for demos
RUN python3 -m venv /opt/mogadishu/python-env

# Copy Python requirements and install
COPY demos/requirements.txt ./python-env/
RUN /opt/mogadishu/python-env/bin/pip install --no-cache-dir -r python-env/requirements.txt

# Copy demo files
COPY demos ./demos/

# Set up directories
RUN mkdir -p /var/lib/mogadishu/data /var/lib/mogadishu/results /var/lib/mogadishu/logs

# Set ownership
RUN chown -R mogadishu:mogadishu /opt/mogadishu /var/lib/mogadishu

# Switch to non-root user
USER mogadishu

# Environment variables
ENV MOGADISHU_HOME=/opt/mogadishu
ENV MOGADISHU_DATA_DIR=/var/lib/mogadishu/data
ENV MOGADISHU_RESULTS_DIR=/var/lib/mogadishu/results
ENV MOGADISHU_LOG_DIR=/var/lib/mogadishu/logs
ENV PYTHON_ENV=/opt/mogadishu/python-env
ENV PATH="/opt/mogadishu/bin:/opt/mogadishu/python-env/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /opt/mogadishu/bin/mogadishu-cli --version || exit 1

# Expose ports for demo web interfaces
EXPOSE 8000 8080 8888

# Default command
CMD ["/opt/mogadishu/bin/mogadishu-cli", "--help"]

# Metadata
LABEL org.opencontainers.image.title="Mogadishu S-Entropy Framework"
LABEL org.opencontainers.image.description="Revolutionary bioreactor modeling through S-entropy navigation"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.authors="Kundai Farai Sachikonye <kundai.sachikonye@wzw.tum.de>"
LABEL org.opencontainers.image.source="https://github.com/fullscreen-triangle/mogadishu"
LABEL org.opencontainers.image.licenses="MIT"
