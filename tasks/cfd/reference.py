import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numba import njit

# inspired by, but not copied from, https://github.com/pmocz/latticeboltzmann-python/blob/main/latticeboltzmann.py

@njit
def calculate_f_eq(rho, ux, uy, cxs, cys, weights):
    """Calculate equilibrium distribution"""
    Nx = rho.shape[1]
    Ny = rho.shape[0]
    NL = len(weights)
    f_eq = np.zeros((Ny, Nx, NL))

    for i in range(NL):
        cx = cxs[i]
        cy = cys[i]
        w = weights[i]
        for y in range(Ny):
            for x in range(Nx):
                cu = cx * ux[y, x] + cy * uy[y, x]
                usqr = ux[y, x] ** 2 + uy[y, x] ** 2
                f_eq[y, x, i] = (
                    rho[y, x] * w * (1 + 3 * cu + 9 * cu**2 / 2 - 3 * usqr / 2)
                )
    return f_eq


@njit
def apply_drift(fluid, cxs, cys):
    """Apply drift step"""
    Ny, Nx, NL = fluid.shape
    F_new = np.zeros_like(fluid)

    for i in range(NL):
        cx = cxs[i]
        cy = cys[i]
        for y in range(Ny):
            for x in range(Nx):
                # Periodic boundary conditions
                xp = (x - cx) % Nx
                yp = (y - cy) % Ny
                F_new[y, x, i] = fluid[yp, xp, i]
    return F_new


@njit
def calculate_macroscopic(fluid, cxs, cys):
    """Calculate macroscopic variables"""
    rho = np.sum(fluid, axis=2)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)

    for i in range(len(cxs)):
        ux += cxs[i] * fluid[:, :, i]
        uy += cys[i] * fluid[:, :, i]

    ux = ux / rho
    uy = uy / rho

    return rho, ux, uy

def simulation_step(
    fluid: npt.NDArray[np.float32],
    cxs: npt.NDArray[np.int32],
    cys: npt.NDArray[np.int32],
    weights: npt.NDArray[np.float32],
    tau: float,
    obstacle: npt.NDArray[np.bool_]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # Drift
    fluid = apply_drift(fluid, cxs, cys)

    # Set reflective boundaries
    bndryF = fluid[obstacle, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    # Calculate fluid variables
    rho, ux, uy = calculate_macroscopic(fluid, cxs, cys)

    # Apply Collision
    f_eq = calculate_f_eq(rho, ux, uy, cxs, cys, weights)
    fluid += -(1.0 / tau) * (fluid - f_eq)

    # # Apply boundary
    fluid[obstacle, :] = bndryF

    return fluid, ux, uy

@njit
def calculate_vorticity(ux, uy):
    """Calculate vorticity field"""
    Ny, Nx = ux.shape
    vorticity = np.zeros_like(ux)

    for y in range(1, Ny - 1):
        for x in range(1, Nx - 1):
            vorticity[y, x] = (ux[y + 1, x] - ux[y - 1, x]) - (
                uy[y, x + 1] - uy[y, x - 1]
            )

    return vorticity