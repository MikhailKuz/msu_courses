## Description
The functionality is divided into two parts: working with datasets and with models.  
First, the required data is loaded. If necessary, unnecessary features can be omitted during further work.  
After that, you can select or train a model, or view information about the model.  
Note that the plot in the information model's section is updated only when you manually refresh the tab.

## Docker
* To pull docker image: docker pull mikkuz/flask_server
* To build docker image: docker build -t mikkuz/flask_server . --no-cache
* To run docker image: docker run -p 5000:5000 -v "$PWD/:/root/server" --rm -i mikkuz/flask_server
