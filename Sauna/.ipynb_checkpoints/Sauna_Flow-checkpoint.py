import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Real parameters
Hgt = 2
g   = 9.8
Ra  = 2e8
Pr  = 0.7
Rγ  = 2e6
cp  = 718
ΔT  = 1
Sm = 1
τ  = 5e-5
η  = 50                   # Step function parameter
α  = 3
β  = g * Hgt / (cp * ΔT)
sqRγ = np.sqrt(Rγ)
sqRγPr = np.sqrt(Rγ * Pr)
sqRγPrSm = np.sqrt(Rγ * Pr * Sm)
PrRa = Pr * Ra
γ  = 0.19

# Parameters
Lx, Lz = 1, 1
Nx, Nz = 64, 64
dealias = 3/2
stop_sim_time = 40
timestepper = d3.RK222
max_timestep = 0.01
dtype = np.float64

# Basis nodes and coordinates
coords = d3.CartesianCoordinates('x', 'z')
dist   = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT( coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name = 'p', bases = (xbasis, zbasis))                # Pressure
b = dist.Field(name = 'b', bases = (xbasis, zbasis))                # Buoyancy
q = dist.Field(name = 'q', bases = (xbasis, zbasis))                # Specific humidity
u = dist.VectorField(coords, name = 'u', bases = (xbasis, zbasis))  # Velocity

# Tau polynomials
τp  = dist.Field(name = 'τp')
τb1 = dist.Field(name = 'τb1', bases = xbasis)
τb2 = dist.Field(name = 'τb2', bases = xbasis)
τq1 = dist.Field(name = 'τq1', bases = xbasis)
τq2 = dist.Field(name = 'τq2', bases = xbasis)
τu1 = dist.VectorField(coords, name = 'τu1', bases = xbasis)
τu2 = dist.VectorField(coords, name = 'τu2', bases = xbasis)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis) # Spatial nodes
x̂, ẑ = coords.unit_vector_fields(dist)  # Unit vectors
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ẑ * lift(τu1)
grad_b = d3.grad(b) + ẑ * lift(τb1)
grad_q = d3.grad(q) + ẑ * lift(τq1)
Δu = d3.div(grad_u) + lift(τu2)
Δb = d3.div(grad_b) + lift(τb2)
Δq = d3.div(grad_q) + lift(τq2)

# Heaviside function setup
ztmp = dist.Field(bases = zbasis)
ztmp['g'] = z
T = b - β * ztmp
qs = np.exp(α * T)
H  = (1 + np.tanh(η * (q - qs))) / 2
qH = (q - qs) / τ * H

# Problem setup
RainyBenard = d3.IVP([p, b, q, u, τp, τb1, τb2, τq1, τq2, τu1, τu2], namespace = locals())

# DEs
# Convection timescale
#RainyBenard.add_equation("dt(u) - Pr * Δu + grad(p) - PrRa * b * ẑ = -u @ grad(u)") # Momentum
#RainyBenard.add_equation("dt(b) -      Δb = -γ * qH - u @ grad(b)")                 # Buoyancy
#RainyBenard.add_equation("dt(q) - Sm * Δq =      qH - u @ grad(q)")                 # Humidity
#RainyBenard.add_equation("trace(grad_u) + τp = 0")                                  # Incompressible
#RainyBenard.add_equation("integ(p) = 0")                                            # Gauge pressure

# Humidity time scal
RainyBenard.add_equation("dt(u) - (Pr /  sqRγ)   * Δu + grad(p) - b * ẑ = -u @ grad(u)") # Momentum
RainyBenard.add_equation("dt(b) - (1 /   sqRγPr) * Δb = γ * qH - u @ grad(b)")           # Buoyancy
RainyBenard.add_equation("dt(q) - (1 / sqRγPrSm) * Δq =   - qH - u @ grad(q)")           # Humidity
RainyBenard.add_equation("trace(grad_u) + τp = 0")                                       # Incompressible
RainyBenard.add_equation("integ(p) = 0")                                                 # Gauge pressure

# BCs
# No slip/penetration conditions
RainyBenard.add_equation("u(z = 0)  = 0")    # Bottom
RainyBenard.add_equation("u(z = Lz) = 0")    # Top

RainyBenard.add_equation("ẑ @ grad(b)(z = 0)  = 0")
#RainyBenard.add_equation("b(z = 0)  = 1")
RainyBenard.add_equation("ẑ @ grad(b)(z = Lz) = 0")

RainyBenard.add_equation("ẑ @ grad(q)(z = 0)  = 0")
#RainyBenard.add_equation("q(z = 0)  = .1")
RainyBenard.add_equation("ẑ @ grad(q)(z = Lz) = 0")

# Solver
solver = RainyBenard.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# ICs
stp = lambda A: (1 + np.tanh(20 * A)) / 2                      # Step function

b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z)                                         # Damp noise at walls
b['g'] += 5 * (stp(x - Lx * 7 / 16) - stp(x - Lx * 9 / 16)) * (1 - stp(z - Lz / 4))
b['g'] += 1

q.fill_random('g', seed=210, distribution='normal', scale=1e-3) # Random noise
q['g'] *= z * (Lz - z)
q['g'] += 900 * (stp(x - Lx * 7 / 16) - stp(x - Lx * 9 / 16)) * (1 - stp(z - Lz / 4))

# CFL conditions
CFL = d3.CFL(solver, initial_dt=max_timestep * 1e-3, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.25, max_dt=max_timestep)
CFL.add_velocity(u)

# Setup recording
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.05, max_writes=50)
snapshots.add_task(T, name='Temperature')
snapshots.add_task(100 * q / qs, name='Rel. Humidity')
snapshots.add_task(np.sqrt(u @ u), name='Speed')
snapshots.add_task(-d3.div(d3.skew(u)), name='Vorticity')

# Run solver
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()