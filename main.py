import abc
import dataclasses

import matplotlib.pyplot as plt
import numpy as np


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

    def simulate(self, dt: float, duration: float, ax):
        iterations = int(duration / dt)
        x = np.empty([iterations])
        y = np.empty_like(x)
        t = 0
        ax.scatter(*self.particle.position, marker='x')
        for i in range(iterations):
            total_force = sum(map(lambda force: force.at(self.particle, t), self.forces))
            acceleration = total_force / self.particle.mass
            self.particle.velocity += acceleration * dt
            self.particle.position += self.particle.velocity * dt
            x[i], y[i] = self.particle.position
            t += dt
        ax.scatter(*self.particle.position, marker='x')
        ax.plot(x, y)

    def visualize(self, field: Force, ax):
        X = np.arange(*ax.get_xlim(), (max(ax.get_xlim()) - min(ax.get_xlim())) / 10)
        Y = np.arange(*ax.get_ylim(), (max(ax.get_ylim()) - min(ax.get_ylim())) / 10)
        U, V = np.meshgrid(X, Y)
        Ex, Ey = field.at(Particle(0, self.particle.charge, 0, (U, V)), 0)
        ax.quiver(X, Y, Ex, -Ey)  # Note: quiver reverses y direction.


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
    trap_size = 0.01
    initial_position = np.array([-0.01, 0.005])  # Trap size is no the order of 1 cm (0.01 m)
    initial_velocity = np.array([0.0, -0.0])
    pollen_mass = calculate_particle_mass()
    pollen_charge = 1e-14  # 7.359e-15 derived from q = g*m/E0, our calculation resulted in 5.11e-15
    Vac_frequency = 50.0  # 50 Hz from outlet
    omega = 2 * np.pi * Vac_frequency
    Vac = 6000  # 0-6 kV
    gamma = 1620  # 1620 Hz From IonTrapPhysics pdf
    # Zeff is a constant that depends on the geometry of the trap, and we expect it's value to be comparable to the
    # spacing between the trap electrodes. We will assume 1 cm which is 0.01 m.
    # Zeff = 0.0079192
    Zeff = 0.01 * 0.15
    A2 = (Vac / Zeff ** 2) / (-4)
    forces = [
        ElectricForce(A2, omega),
        DragForce(gamma),
        # GravitationalForce(g=9.8),
        # ElectrostaticForce(E0=5000)
    ]
    particle = Particle(pollen_mass, pollen_charge, initial_velocity, initial_position)

    if Vac < (particle.mass * omega * gamma * Zeff ** 2 / particle.charge):
        print("System is stable")
    else:
        print("System is not stable")

    fig, ax = plt.subplots()
    ax.set_xlim(-trap_size, trap_size)
    ax.set_ylim(-trap_size, trap_size)

    simulation = Simulation(particle, forces)
    simulation.simulate(0.00005, 1, ax)
    simulation.visualize(ElectricForce(A2, omega), ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == '__main__':
    main()