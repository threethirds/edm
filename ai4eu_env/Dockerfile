FROM python:3.9

RUN mkdir -p ~/app
WORKDIR ~/app

COPY requirements.txt requirements.txt
RUN pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

COPY ai4eu_env/model.proto model.proto
COPY ai4eu_env/service.py service.py
COPY setup.py setup.py
COPY agent/ agent/
COPY env/ env/

RUN pip install -e .

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model.proto

RUN useradd app
USER app

EXPOSE 8061

CMD python service.py
