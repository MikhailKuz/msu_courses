FROM python:3.8-slim

COPY server/src/requirements.txt /root/server/src/requirements.txt

RUN chown -R root:root /root/server

WORKDIR /root/server/
RUN pip3 install -r requirements.txt

COPY server/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

WORKDIR /root/server/src
RUN chmod +x run.py
CMD ["python3", "run.py"]
