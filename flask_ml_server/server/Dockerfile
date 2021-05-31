FROM python:3.8-slim

COPY . /root/

RUN chown -R root:root /root/

WORKDIR /root/
RUN pip3 install -r ./src/requirements.txt

RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

WORKDIR /root/src
RUN chmod +x run.py
CMD ["python3", "run.py"]
