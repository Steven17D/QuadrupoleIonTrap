import dataclasses

import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class Field:
    def __init__(self, A2: float, omega: float):
        self.omega = omega
        self.A2 = A2

    def at(self, t, position: np.array):
        x, y = position
        coefficient = 2 * self.A2 * np.cos(self.omega * t)
        # return coefficient * np.array([(-1) * y, -x])
        return coefficient * np.array([(-1) * x, y])


@dataclasses.dataclass
class Particle:
    mass: float
    charge: float
    velocity: np.array
    position: np.array


def simulate(field: Field, particle: Particle, gamma: float, dt: float, duration: float, ax):
    iterations = int(duration / dt)
    x = np.empty([iterations])
    y = np.empty_like(x)
    t = 0
    g = 9.8
    E0 = 5000
    # gravitational_force = g * particle.mass * np.array([0., -1.])
    # electrostatic_force = E0 * particle.charge * np.array([0., 1.])
    ax.scatter(*particle.position, marker='x')
    for i in range(iterations):
        field_force = field.at(t, particle.position) * particle.charge
        drag_force = - (gamma * particle.mass) * particle.velocity
        total_force = field_force + drag_force  # + gravitational_force + electrostatic_force
        acceleration = total_force / particle.mass
        particle.velocity += acceleration * dt
        particle.position += particle.velocity * dt
        x[i], y[i] = particle.position
        t += dt
    ax.scatter(*particle.position, marker='x')
    ax.plot(x, y)


def visualize(field: Field, ax):
    X = np.arange(*ax.get_xlim(), (max(ax.get_xlim()) - min(ax.get_xlim())) / 10)
    Y = np.arange(*ax.get_ylim(), (max(ax.get_ylim()) - min(ax.get_ylim())) / 10)
    U, V = np.meshgrid(X, Y)
    Ex, Ey = field.at(0, (U, V))
    ax.quiver(X, Y, Ex, -Ey)  # Note: quiver reverses y direction.


def calculate_particle_mass():
    """
    According to Newtonian Labs document - InstrumentDescriptionEIT.pdf.
    Note: Our measurement shows 3.379e-13 kg
    :return:
    """
    DENSITY = 510  # 510+-40 kg/m^3
    RADIUS = 26e-6  # 26+-2.5 um
    volume = (4/3.0) * np.pi * RADIUS**3
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
    A2 = (Vac / Zeff**2) / (-4)
    field = Field(A2, omega)
    particle = Particle(pollen_mass, pollen_charge, initial_velocity, initial_position)

    if Vac < (particle.mass * omega * gamma * Zeff**2 / particle.charge):
        print("System is stable")
    else:
        print("System is not stable")

    fig, ax = plt.subplots()
    ax.set_xlim(-trap_size, trap_size)
    ax.set_ylim(-trap_size, trap_size)

    simulate(field, particle, gamma, 0.00005, 1, ax)
    visualize(field, ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == '__main__':
    main()
