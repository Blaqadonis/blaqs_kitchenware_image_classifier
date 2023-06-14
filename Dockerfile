FROM python:3.9-slim-buster

RUN pip install -U pip

WORKDIR /app

COPY [ "local_server/classifier_predict.py", "xception_v4_large_12_0.965.h5", "mlflow_server/classifier_predict_mlflow.py", "requirements.txt", "./" ]

RUN pip install -r requirements.txt

EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:9696", "classifier_predict:app" ]