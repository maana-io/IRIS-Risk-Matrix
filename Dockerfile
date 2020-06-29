#FROM tiangolo/uvicorn-gunicorn:python3.7-alpine3.8
FROM tiangolo/uvicorn-gunicorn:python3.7

RUN pip3 install --upgrade pip

WORKDIR /

RUN rm -rf /app
COPY ./app /app

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

#COPY .env /.env
COPY gunicorn_conf.py /gunicorn_conf.py
COPY start.sh /start.sh
COPY start-reload.sh /start-reload.sh

EXPOSE 8050

#RUN cd ./test/ && export PYTHONPATH=./../ && python3.7 helpers.py

CMD /start.sh
