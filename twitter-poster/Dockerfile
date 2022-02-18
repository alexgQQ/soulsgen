FROM acidrain/python-poetry:3.9-slim

COPY . /app
WORKDIR /app
RUN poetry export -f requirements.txt --output requirements.txt && \
    pip install -r requirements.txt

CMD ["python", "twitter/main.py", "post"]
