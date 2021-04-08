import gym
from numba import jit
import numpy as np


@jit(nopython=True)
def ionisation(d):
    return 0.01 * np.exp(4.6 * d) if d < 2 else np.inf


@jit(nopython=True)
def voltage(d):
    return 6.9 * np.exp(1.3 * d)


def jit_spark(Δt, d, debris, sparking_gap, spark_duration, zrr_at_30V):

    voltages = []
    Δz = 0

    while Δt > 0:

        d_noisy = d / sparking_gap + 0.1 * np.random.rand()

        # Time for a spark to form
        if d_noisy > 2:  # infinity, never fires
            break
        else:
            Δt_ionisation = ionisation(d_noisy)

        # Voltage of the spark
        spark_voltage = voltage(d_noisy)

        # Duration of the spark (constant in this model)
        spark_duration = spark_duration

        # material removed
        Δz_energy = zrr_at_30V * spark_voltage * spark_duration / 30 / 10 ** 6
        Δz_material = max(0, 1 - debris / 100) * Δz_energy

        # update state
        Δt -= (Δt_ionisation + spark_duration)
        voltages.append(spark_voltage)
        d += Δz_material
        debris += Δz_material
        Δz += Δz_material

    return voltages, Δz


class NumpyEDM1(gym.Env):
    """
    Drills a hole of a given area

    clear distance    100 mm
    sparking gap      100 μm
    control distance   10 μm
    control every     100 ms
    ON + OFF duration 100 μs
    mrr_at_30V          1 mm^3/s

    """
    sparking_gap = 100  # μm
    control_distance = 10  # μm
    control_every = 100  # ms
    spark_duration = 50  # μs
    mrr_at_30V = 1  # mm^3/s

    def __init__(self, area):
        self.area = area  # mm^2
        self.zrr_at_30V = self.mrr_at_30V / area

        self.z_electrode = None  # μm
        self.z_material = None  # μm
        self.debris = None  # μm
        self.t = None

    def spark(self, Δt=100_000):

        d = self.z_electrode - self.z_material

        voltages, Δz = jit_spark(Δt=Δt,
                                 d=d,
                                 debris=self.debris,
                                 sparking_gap=self.sparking_gap,
                                 spark_duration=self.spark_duration,
                                 zrr_at_30V=self.zrr_at_30V)

        self.z_material -= Δz
        self.debris += Δz

        if voltages:
            return np.array([np.mean(voltages), len(voltages) / 1000])
        else:
            return np.array([100, 0])

    def step(self, action):

        end = np.bool8(False)
        reward = np.float32(0)

        if action == 0:
            pass
        elif action == 1:
            self.z_electrode -= self.control_distance
            reward = np.float32(1)
        elif action == 2:
            self.z_electrode += self.control_distance
            reward = np.float32(-1)
        elif action == 3:
            self.debris = 0
            self.t += 10 ** 6

        if self.z_electrode <= self.z_material:
            end = np.bool8(True)
            reward = np.float32(-10)

        sparks = self.spark()

        if self.t > 10 ** 9:
            end = np.bool8(True)

        return sparks, reward, end, {}

    def reset(self):
        self.z_electrode = 0  # μm
        self.z_material = -100  # μm
        self.debris = 0  # μm
        self.t = 0

        sparks = self.spark()

        return sparks

    def render(self, mode='human'):
        pass
