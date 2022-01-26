ARG BASE_IMAGE_TAG=3.9.0

FROM python:${BASE_IMAGE_TAG}

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
	&& apt-get install -y software-properties-common \
	&& echo -e '\n' | add-apt-repository ppa:ubuntugis/ppa \
	&& apt-get install -y gdal-bin \
	&& apt-get install -y python3-gdal \
    && apt-get install -y --no-install-recommends \
        postgresql-client \

# install dependencies
COPY requirements.txt /usr/src/app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN python manage.py runserver

EXPOSE 8000