from concurrent import futures

import grpc

from agent.edm import stub
import model_pb2
import model_pb2_grpc


class ModelServicer(model_pb2_grpc.ModelServicer):

    def __init__(self):
        self.agent = stub()

    def reset(self, request, context):
        _ = request
        _ = context

        # voltage, sparks = self.agent.reset()

        # return model_pb2.Observation(voltage=float(voltage),
        #                              sparks=float(sparks))
        return model_pb2.Init()

    def step(self, request, context):
        _ = context

        # obs, reward, end, _ = self.env.step(request.action)
        # voltage, sparks = obs

        # return model_pb2.Observation(voltage=float(voltage),
        #                              sparks=float(sparks))

        return model_pb2.Action(action=1)


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
model_pb2_grpc.add_ModelServicer_to_server(ModelServicer(), server)
server.add_insecure_port('[::]:8061')
server.start()
server.wait_for_termination()
