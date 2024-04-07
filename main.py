import abc
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.widgets import Slider


@dataclasses.dataclass
class Particle:
    mass: float
    charge: float
    velocity: np.array
    position: np.array


class Force(abc.ABC):
    @abc.abstractmethod
    def at(self, particle: Particle, time: float) -> np.array:
        pass


class ElectricForce(Force):
    def __init__(self, A2: float, omega: float):
        self.omega = omega
        self.A2 = A2

    def at(self, particle: Particle, time: float):
        x, y = particle.position
        coefficient = 2 * self.A2 * np.cos(self.omega * time)
        return coefficient * np.array([(-1) * x, y]) * particle.charge


class DragForce(Force):
    def __init__(self, gamma: float):
        self.gamma = gamma

    def at(self, particle: Particle, time: float):
        return - (self.gamma * particle.mass) * particle.velocity


class GravitationalForce(Force):
    def __init__(self, g: float):
        self.g = g

    def at(self, particle: Particle, time: float):
        return self.g * particle.mass * np.array([0., -1.])


class ElectrostaticForce(Force):
    def __init__(self, E0: float):
        self.E0 = E0

    def at(self, particle: Particle, time: float):
        return self.E0 * particle.charge * np.array([0., 1.])


class Simulation:
    def __init__(self, particle: Particle, forces: list[Force]):
        self.particle = particle
        self.forces = forces

    def simulate(self, dt: float, duration: float, plot, writer: FFMpegWriter = None, time_factor: int = 2):
        iterations = int(duration / dt)
        x = np.empty([iterations])
        y = np.empty_like(x)
        t = 0
        for i in range(iterations):
            total_force = sum(map(lambda force: force.at(self.particle, t), self.forces))
            acceleration = total_force / self.particle.mass
            self.particle.velocity += acceleration * dt
            self.particle.position += self.particle.velocity * dt
            x[i], y[i] = self.particle.position
            t += dt
            if writer is not None and (i % (iterations // (writer.fps * time_factor))) == 0:
                plot.set_data(x[:i], y[:i])
                writer.grab_frame()
        plot.set_data(x, y)

    def visualize_field(self, field: Force, ax, time: float = 0):
        X = np.arange(*ax.get_xlim(), (max(ax.get_xlim()) - min(ax.get_xlim())) / 10)  # TODO: Use linspace
        Y = np.arange(*ax.get_ylim(), (max(ax.get_ylim()) - min(ax.get_ylim())) / 10)
        U, V = np.meshgrid(X, Y)
        Ex, Ey = field.at(Particle(0, self.particle.charge, 0, (U, V)), 0)
        ax.quiver(X, Y, Ex, Ey)  # Note: quiver reverses y direction.


def calculate_particle_mass():
    """
    According to Newtonian Labs document - InstrumentDescriptionEIT.pdf.
    Note: Our measurement shows 3.379e-13 kg
    :return:
    """
    DENSITY = 510  # 510+-40 kg/m^3
    RADIUS = 26e-6  # 26+-2.5 um
    volume = (4 / 3.0) * np.pi * RADIUS ** 3
    mass = DENSITY * volume
    return mass


def main():
    simulation, A2, Vac, omega = create_simulation()

    fig, ax = plt.subplots()
    # plt.title(fr"$V_{{ac}} = {Vac}, \Gamma = {gamma}, Z_{{eff}} = {Zeff}, \omega=2\pi/{Vac_frequency}$")

    trap_size = 0.01
    ax.set_xlim(-trap_size, trap_size)
    ax.set_ylim(-trap_size, trap_size)

    simulation.visualize_field(ElectricForce(A2, omega), ax)
    plot, = plt.plot([], [])
    record = False
    if record:
        metadata = dict(title='Movie', artist='codinglikemad')
        writer = FFMpegWriter(fps=60, metadata=metadata)
        with writer.saving(fig, 'Movie.mp4', 100):
            simulation.simulate(0.00005, 2, plot, writer, time_factor=2)
            writer.grab_frame()
    else:
        simulation.simulate(0.00005, 1, plot)
        Vac_slider = Slider(plt.axes([0.25, 0, 0.65, 0.03]), '$V_{ac}$', 0, 6000, valinit=Vac, valstep=200)

        def update_Vac(val):
            print("A")
            simulation, A2, Vac, omega = create_simulation(Vac=val)
            simulation.simulate(0.00005, 1, plot)
            print("B")
            fig.canvas.draw_idle()
            print("C")

        Vac_slider.on_changed(update_Vac)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def create_simulation(Vac: float = 3000):
    initial_position = np.array([-0.003, 0.003])  # Trap size is no the order of 1 cm (0.01 m)
    initial_velocity = np.array([-0.2, -0.1])
    pollen_mass = calculate_particle_mass()
    pollen_charge = 2e-14  # 7.359e-15 derived from q = g*m/E0, our calculation resulted in 5.11e-15
    Vac_frequency = 50.0  # 50 Hz from outlet
    omega = 2 * np.pi * Vac_frequency
    # Vac = 3000  # 0-6 kV
    gamma = 0.02 * 1620  # 1620 Hz From IonTrapPhysics pdf
    # Zeff is a constant that depends on the geometry of the trap, and we expect it's value to be comparable to the
    # spacing between the trap electrodes. We will assume 1 cm which is 0.01 m.
    Zeff = 0.01 * 0.5
    A2 = (Vac / Zeff ** 2) / (-4)
    forces = [
        ElectricForce(A2, omega),
        DragForce(gamma),
        GravitationalForce(g=9.8),
        ElectrostaticForce(E0=5000)
    ]
    particle = Particle(pollen_mass, pollen_charge, initial_velocity, initial_position)
    if Vac < (particle.mass * omega * gamma * (Zeff ** 2) / particle.charge):
        print("System is stable")
    else:
        print("System is not stable")
    simulation = Simulation(particle, forces)
    return simulation, A2, Vac, omega


if __name__ == '__main__':
    main()
