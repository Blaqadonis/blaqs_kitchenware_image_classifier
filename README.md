# blaqs_kitchenware_image_classifier
![a](https://github.com/Blaqadonis/blaqs_kitchenware_image_classifier/assets/100685852/9c345cd0-b6b9-46b1-8981-4a8b0014714c)

DataTalks.Club organized an image classification competition for their 2022 cohort. I participated in this competition. 

With more than 9000 photos of kitchen utensils, my model was able to attain an accuracy of 93% on the validation dataset. I think I ended up top 20 in the competition.

Anyway, I retrained my model and added some augmentation here and there. Now I have a model with 96% accuracy on validation set. I could not upload the new  training  script here - the tensorflow model is quite heavy!

In this competition we were asked to classify images of different kitchenware items into 6 classes:

```cups```
```glasses```
```plates```
```spoons```
```forks```
```knives```

This is a link to the competition for more details https://www.kaggle.com/competitions/kitchenware-classification

## How it works!

Everything here runs locally. If you want to try out the service, follow the steps below:

Before you proceed, create a virtual environment. I used ```python version 3.9``` 

To create an environment with that version of python using Conda: ```conda create -n <env-name> python=3.9```

Just replace <env-name> with any title you want. Next:

 conda activate <env-name>

to activate the environment.

1. Running the container (Dockerfile)
First, you need to have docker installed on your system. I am using a windows machine, and I have docker desktop installed on my system. If you do not have that, then you should try doing that first. If you are all set and good, then proceed.

Now run:

 pip install -r requirements.txt

to install all necessary external dependencies.

Next, Run:

docker build -t <service-name>:v1 .

Replace <service-name> with whatever name you wish to give to the body fat percent estimator service, to build the image.

To run this service:

docker run -it --rm -p 9696:9696 <service-name>:latest

NOTE: I am running this on Windows hence Waitress. If your local machine requires Gunicorn, I think the Dockerfile should be edited with something like this:


RUN pip install -U pip

WORKDIR /app

COPY [ "online_webservice_flask/predict.py", "models/pipeline.bin", "requirements.txt", "./" ]

RUN pip install -r requirements.txt

EXPOSE 9696 
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
If the container is up and running, open up a new terminal. Reactivate the Conda environment. Run:

python test.py

NOTE: test.py is an example of data you can send to the ENTRYPOINT to interact with the service. Edit it as much as you desire and try out some predictions.

## 2. Simple web service (server managed locally with Flask)
  
You need to first run:

```pip install -r requirements.txt```

Followed by ```python local_server/classifier_predict.py``` to run this service.

Open up a new terminal. Run ```python test.py``` to interact with the service.

## 3. Web service hosted and managed on MLflow servers

  You need to first run: ```pip install -r requirements.txt```

Next, spin up the MLflow server with: ```mlflow server --backend-store-uri sqlite:///local_server.db --default-artifact-root ./artifacts --host localhost --port 5000```

This will create a folder ```artifacts``` on your local machine, as well as the database ```local_server```.

Now, run ```python mlflow_server/classifier_predict_mlflow.py``` followed by ```python test.py```

Try it out with family, friends, colleagues, neighbours, and let me know how to improve on it.
