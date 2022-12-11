FROM python:3.9

RUN echo 'Launched build!'
WORKDIR /usr/src/app

RUN echo 'Copying project files'
ADD swagger_api /usr/src/app/swagger_api
COPY README.md /usr/src/app/
COPY run_api.py /usr/src/app/
RUN chmod -R 777 /usr/src/app/

RUN echo 'Copying req files'
COPY requirements.txt /usr/src/app/

RUN echo 'Installing dependencies'
RUN pip install -r requirements.txt

RUN echo 'Exposing ports'
EXPOSE 4000

RUN echo 'Adding run command'
ENV RUNTIME_DOCKER Yes
CMD python run_api.py

