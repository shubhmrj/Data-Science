FROM python:3.9-slim-bullseye

WORKDIR /app
COPY . /app

RUN apt-get update -y && apt-get install -y --no-install-recommends awscli \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

CMD [ "python3", "application.py" ]