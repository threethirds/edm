from concurrent import futures

import grpc

from env.edm import NumpyEDM1
import model_pb2
import model_pb2_grpc


class ModelServicer(model_pb2_grpc.ModelServicer):

    def __init__(self):
        self.env = NumpyEDM1(1)

    def reset(self, request, context):
        _ = request
        _ = context

        voltage, sparks = self.env.reset()

        return model_pb2.Observation(voltage=float(voltage),
                                     sparks=float(sparks))

    def step(self, request, context):
        _ = context

        obs, reward, end, _ = self.env.step(request.action)
        voltage, sparks = obs

        return model_pb2.Observation(voltage=float(voltage),
                                     sparks=float(sparks))


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
model_pb2_grpc.add_ModelServicer_to_server(ModelServicer(), server)
server.add_insecure_port('[::]:8001')
server.start()
server.wait_for_termination()
