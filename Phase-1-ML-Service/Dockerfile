
FROM python:3.12-slim

WORKDIR /app

# Install uv to install the locked dependencies.
RUN pip install --no-cache-dir uv

# Install the exact dependencies from the lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the API and the trained model artifacts.
COPY app.py ./
COPY model/ ./model/

# Flask listens on port 5001 in this project.
EXPOSE 5001

CMD [".venv/bin/python", "app.py"]
