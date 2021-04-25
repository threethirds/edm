FROM python:3.9

RUN mkdir -p ~/app
WORKDIR ~/app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#COPY ai4eu_agent/model.proto model.proto
COPY ai4eu_env/model.proto model.proto
COPY ai4eu_env/client.py client.py

COPY ai4eu_agent/service.py service.py
COPY setup.py setup.py
COPY agent/ agent/
COPY env/ env/

RUN pip install -e .

COPY agent_weights.pt agent_weights.pt
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model.proto

RUN useradd app
USER app

EXPOSE 8061
EXPOSE 8062

CMD bokeh serve \
    --allow-websocket-origin="*" \
    --port 8062 \
    --address 0.0.0.0 \
    service.py