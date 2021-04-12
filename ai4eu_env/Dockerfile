FROM python:3.9

COPY ai4eu_env/model.proto model.proto
COPY ai4eu_env/service.py service.py
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY env/ env/

RUN pip install -r requirements.txt
RUN pip install -e .

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model.proto

RUN useradd app
USER app

CMD [ "python", "service.py" ]
