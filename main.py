import abc
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.widgets import Slider


DENSITY = 510  # 510+-40 kg/m^3
RADIUS = 26e-6 / 2.0  # The diameter 26+-2.5 um


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
    def at(self, particle: Particle, time: float):
        mu = 1.8e-5  # Air dynamic viscosity in kg / (m * sec)
        return -6 * np.pi * mu * RADIUS * particle.velocity


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

    def visualize_field(self, field: Force, ax, t: float = 0):
        X = np.linspace(*ax.get_xlim(), 9)
        Y = np.linspace(*ax.get_ylim(), 9)
        U, V = np.meshgrid(X, Y)
        Ex, Ey = field.at(Particle(t, self.particle.charge, 0, (U, V)), 0)
        ax.quiver(X, Y, Ex, Ey)  # Note: quiver reverses y direction.


def calculate_particle_mass():
    """
    According to Newtonian Labs document - InstrumentDescriptionEIT.pdf.
    Note: Our measurement shows 3.379e-13 kg
    :return:
    """
    volume = (4 / 3.0) * np.pi * RADIUS ** 3
    mass = DENSITY * volume
    return mass

@dataclasses.dataclass
class Configuration:
    Vac: float = 3000  # 0-6 kV
    Zeff: float = 1.16e-3  # Derived from our measurements
    Vac_frequency: float = 50.0  # 50 Hz from outlet
    g: float = 9.8
    E0: float = 5000.
    x0: float = -0.002
    y0: float = 0.002
    vx0: float = 0.
    vy0: float = 0.
    pollen_charge: float = 7.359e-15  # 7.359e-15 derived from q = g*m/E0, our calculation resulted in 5.11e-15
    pollen_mass: float = calculate_particle_mass()

    def __post_init__(self):
        self.E0 = self.pollen_mass * self.g / self.pollen_charge  # Set


def create_simulation(ax,
                      Vac: float,
                      Zeff: float,
                      Vac_frequency: float,
                      g: float,
                      E0: float,
                      x0: float,
                      y0: float,
                      vx0: float,
                      vy0: float,
                      pollen_charge: float,
                      pollen_mass: float,
):
    initial_position = np.array([x0, y0])
    initial_velocity = np.array([vx0, vy0])
    # Zeff is a constant that depends on the geometry of the trap, and we expect it's value to be comparable to the
    # spacing between the trap electrodes. We will assume 1 cm which is 0.01 m.
    omega = 2 * np.pi * Vac_frequency
    A2 = (Vac / Zeff ** 2) / (-4)
    forces = [ElectricForce(A2, omega), DragForce(), ElectrostaticForce(E0), GravitationalForce(g)]
    particle = Particle(pollen_mass, pollen_charge, initial_velocity, initial_position)
    simulation = Simulation(particle, forces)
    simulation.visualize_field(ElectricForce(A2, omega), ax)
    plt.suptitle(fr"Paul trap simulation")
    return simulation


def main():
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_axes([0.2, 0.3, 0.75, 0.60])
    trap_size = 0.01
    ax.set_xlim(-trap_size*2, trap_size*2)
    ax.set_ylim(-trap_size*2, trap_size*2)
    Vac_axes = fig.add_axes([0.25, 0, 0.65, 0.03])
    Zeff_axes = fig.add_axes([0.25, 0.03, 0.65, 0.03])
    q_axes = fig.add_axes([0.25, 0.06, 0.65, 0.03])
    frequency_axes = fig.add_axes([0.25, 0.09, 0.65, 0.03])
    g_axes = fig.add_axes([0.25, 0.12, 0.65, 0.03])
    E0_axes = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    x0_axes = fig.add_axes([0.25, 0.18, 0.25, 0.03])
    y0_axes = fig.add_axes([0.65, 0.18, 0.25, 0.03])
    vx0_axes = fig.add_axes([0.25, 0.21, 0.25, 0.03])
    vy0_axes = fig.add_axes([0.65, 0.21, 0.25, 0.03])

    configuration = Configuration()
    simulation = create_simulation(ax, **configuration.__dict__)

    plot, = ax.plot([], [])
    record = False
    if record:
        metadata = dict(title='Movie', artist='codinglikemad')
        writer = FFMpegWriter(fps=60, metadata=metadata)
        with writer.saving(fig, 'Movie.mp4', 100):
            simulation.simulate(0.00005, 2, plot, writer, time_factor=2)
            writer.grab_frame()
    else:
        simulation.simulate(0.00005, 1, plot)
        Vac_slider = Slider(Vac_axes, '$V_{ac}$', 0, configuration.Vac * 2, valinit=configuration.Vac, valstep=configuration.Vac / 10)
        Zeff_slider = Slider(Zeff_axes, '$Z_{eff}$', 0, trap_size / 2, valinit=configuration.Zeff, valstep=0.0002)
        frequency_slider = Slider(frequency_axes, r'$V_{ac} frequency$', 0, configuration.Vac_frequency * 2, valinit=configuration.Vac_frequency, valstep=10)
        g_slider = Slider(g_axes, r'$g$', 0, configuration.g * 2, valinit=configuration.g, valstep=0.1)
        E0_slider = Slider(E0_axes, r'$E_0$', 0, configuration.E0 * 2, valinit=configuration.E0, valstep=100)
        x0_slider = Slider(x0_axes, r'$x_0$', -trap_size, trap_size, valinit=configuration.x0, valstep=trap_size / 5)
        y0_slider = Slider(y0_axes, r'$y_0$', -trap_size, trap_size, valinit=configuration.y0, valstep=trap_size / 5)
        vx0_slider = Slider(vx0_axes, r'$vx_0$', -trap_size * 10, trap_size * 10, valinit=configuration.vx0, valstep=trap_size)
        vy0_slider = Slider(vy0_axes, r'$vy_0$', -trap_size * 10, trap_size * 10, valinit=configuration.vy0, valstep=trap_size)
        q_slider = Slider(q_axes, r'Particle charge', 1e-15, 1e-14, valinit=configuration.pollen_charge, valstep=5e-16)

        def update(name: str, value):
            setattr(configuration, name, value)
            simulation = create_simulation(ax, **configuration.__dict__)
            simulation.simulate(0.00005, 2, plot)
            fig.canvas.draw_idle()

        Vac_slider.on_changed(lambda val: update("Vac", val))
        Zeff_slider.on_changed(lambda val: update("Zeff", val))
        frequency_slider.on_changed(lambda val: update("Vac_frequency", val))
        g_slider.on_changed(lambda val: update("g", val))
        E0_slider.on_changed(lambda val: update("E0", val))
        x0_slider.on_changed(lambda val: update("x0", val))
        y0_slider.on_changed(lambda val: update("y0", val))
        vx0_slider.on_changed(lambda val: update("vx0", val))
        vy0_slider.on_changed(lambda val: update("vy0", val))
        q_slider.on_changed(lambda val: update("pollen_charge", val))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == '__main__':
    main()
