from bokeh.layouts import column
from bokeh.models import Button
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc
from bokeh.plotting import figure
import grpc
import torch

from agent.edm import Network as Agent
import model_pb2
import model_pb2_grpc

ds1 = ColumnDataSource(data=dict(x=[], y=[]))
p1 = figure(title="Agent action", plot_width=500, plot_height=200)
r1 = p1.line(x='x', y='y', source=ds1)

ds2 = ColumnDataSource(data=dict(x=[], y=[]))
p2 = figure(title="Observed sparking voltage", plot_width=500, plot_height=200)
r2 = p2.line(x='x', y='y', source=ds2)

ds3 = ColumnDataSource(data=dict(x=[], y=[]))
p3 = figure(title="Observed sparking frequency", plot_width=500, plot_height=200)
r3 = p3.line(x='x', y='y', source=ds3)

agent = Agent(prob_uniform=0.)
agent.load_state_dict(torch.load('agent_weights.pt'))
agent.eval()
agent.running_n = 15937376

channel = grpc.insecure_channel('localhost:8061')
stub = model_pb2_grpc.ModelStub(channel)

button_clicked = False
step = 0
obs = stub.reset(model_pb2.Init())


def periodic_callback():
    global step
    global obs

    if not button_clicked or step > 3000:
        return

    value, a_dist = agent(torch.tensor([obs.voltage, obs.sparks], dtype=torch.float32).unsqueeze(0))
    a = a_dist.sample()
    obs = stub.step(model_pb2.Action(action=a.item()))
    step += 1

    # Action
    ds1.stream({'x': [step], 'y': [a.item()]})
    ds2.stream({'x': [step], 'y': [obs.voltage]})
    ds3.stream({'x': [step], 'y': [obs.sparks]})


def callback():
    global button_clicked
    button_clicked = True


# add a button widget and configure with the call back
button = Button(label="Start control")
button.on_click(callback)

# put the button and plot in a layout and add to the document
bokeh_doc = curdoc()
bokeh_doc.add_root(column(button, p1, p2, p3))
bokeh_doc.add_periodic_callback(periodic_callback, 10)
