FROM acidrain/python-poetry:3.9-slim

COPY . /app
WORKDIR /app
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev
CMD ["twitter", "post"]
