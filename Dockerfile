FROM python:3.7

COPY . /EdrixsRun
WORKDIR /EdrixsRun

RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD ["python", "./edrixs_run.py"]
