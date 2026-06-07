# HomoeoGWAS — CPU image.
# Bundles plink2 + bcftools so the full pipeline (split/VCF -> fit) works.
# Build:  docker build -t homoeogwas:cpu .
# Run:    docker run --rm homoeogwas:cpu demo
#         docker run --rm -v "$PWD":/work -w /work homoeogwas:cpu fit -c run.yaml
FROM python:3.11-slim

ARG PLINK2_URL=https://s3.amazonaws.com/plink2-assets/alpha7/plink2_linux_x86_64_20260425.zip
ARG PLINK2_SHA256=e70a283aefe004122fca3e632ae0b24023a24635f98a8e768ea8d542bbc659a9

RUN apt-get update && apt-get install -y --no-install-recommends \
        bcftools curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "$PLINK2_URL" -o /tmp/plink2.zip \
    && echo "$PLINK2_SHA256  /tmp/plink2.zip" | sha256sum -c - \
    && unzip -q /tmp/plink2.zip -d /usr/local/bin plink2 \
    && chmod +x /usr/local/bin/plink2 \
    && rm /tmp/plink2.zip

# Pass --build-arg PIP_INDEX_URL=<mirror> to install through a faster pip mirror.
ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /opt/homoeogwas
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" .

WORKDIR /work
ENTRYPOINT ["homoeogwas"]
CMD ["--help"]
