* Docker
    1. Чтобы собрать докер образ: docker build -t flask_server .
    2. Чтобы его запустить: docker run -p 5000:5000 -v "$PWD/:/root/server" --rm -i flask_server
