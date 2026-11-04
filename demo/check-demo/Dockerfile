FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pytest==8.4.2
WORKDIR /workspace
COPY checkout_demo/ /workspace/checkout_demo/
COPY tests/ /workspace/tests/
RUN chmod -R a+rX /workspace
CMD ["python", "-m", "pytest", "--tb=native", "-q"]
