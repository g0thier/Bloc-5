FROM python:3.9.5

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./app /code/app
COPY ./app/src /code/app/src

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt install curl -y
RUN curl -fsSL https://get.deta.dev/cli.sh | sh



CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

CMD gunicorn app.main:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker