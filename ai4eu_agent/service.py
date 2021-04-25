from bokeh.layouts import column
from bokeh.models import Button
from bokeh.plotting import curdoc
from bokeh.plotting import figure
import grpc
import torch

from agent.edm import Network as Agent
import model_pb2
import model_pb2_grpc

p1 = figure(plot_width=500, plot_height=200)
r1 = p1.line(x=[], y=[])
ds1 = r1.data_source

p2 = figure(plot_width=500, plot_height=200)
r2 = p2.line(x=[], y=[])
ds2 = r2.data_source

p3 = figure(plot_width=500, plot_height=200)
r3 = p3.line(x=[], y=[])
ds3 = r3.data_source

agent = Agent(prob_uniform=0.)
agent.load_state_dict(torch.load('agent_weights.pt'))
agent.eval()


# create a callback that adds a number in a random location
def callback():
    with grpc.insecure_channel('localhost:8061') as channel:
        stub = model_pb2_grpc.ModelStub(channel)
        obs = stub.reset(model_pb2.Init())

        for step in range(100):
            value, a_dist = agent(torch.tensor([obs.voltage, obs.sparks], dtype=torch.float32).unsqueeze(0))
            a = a_dist.sample()
            obs = stub.step(model_pb2.Action(action=a.item()))

            # Action
            new_data = dict()
            new_data['x'] = ds1.data['x'] + [step]
            new_data['y'] = ds1.data['y'] + [a.item()]
            ds1.data = new_data

            # Voltage
            new_data = dict()
            new_data['x'] = ds2.data['x'] + [step]
            new_data['y'] = ds2.data['y'] + [obs.voltage]
            ds2.data = new_data

            # Sparks
            new_data = dict()
            new_data['x'] = ds3.data['x'] + [step]
            new_data['y'] = ds3.data['y'] + [obs.sparks]
            ds3.data = new_data


# add a button widget and configure with the call back
button = Button(label="Start control")
button.on_click(callback)

# put the button and plot in a layout and add to the document
curdoc().add_root(column(button, p1, p2, p3))
