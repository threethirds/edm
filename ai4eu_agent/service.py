from concurrent import futures

import grpc

import model_pb2
import model_pb2_grpc


# This is a dummy endpoint in necessary to connect components into a pipeline
class ModelServicer(model_pb2_grpc.ModelServicer):

    def reset(self, request, context):
        _ = request
        _ = context

        return model_pb2.Init()

    def step(self, request, context):
        _ = context

        return model_pb2.Action(action=1)


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
model_pb2_grpc.add_ModelServicer_to_server(ModelServicer(), server)
server.add_insecure_port('[::]:8061')
server.start()
server.wait_for_termination()
