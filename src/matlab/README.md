# MATLAB dynamics and PID control

This part of the project studies a four-coordinate 3R+1P mechanism through
open-loop numerical simulation and joint-space PID position control. It
complements the Python kinematic planner by focusing on state evolution,
generalized inputs, feedback, and visual response.

## Numerical model

The scripts use the project model

```math
M_{project}\ddot q+g_{project}(q)=u,
```

where `M_project` is a constant diagonal matrix and `g_project(q)` is the
configuration-dependent expression assembled in the scripts. The implementation
keeps generalized coordinates and inputs as explicit four-element vectors and
updates position and velocity at each simulation step.

For constant acceleration over one step,

```math
q_{k+1}=q_k+\dot q_k\Delta t+\frac{1}{2}\ddot q_k\Delta t^2,
\qquad
\dot q_{k+1}=\dot q_k+\ddot q_k\Delta t.
```

## PID controller

The sampled controller uses

```math
e_k=q_d-q_k,
\qquad
I_k=I_{k-1}+e_k\Delta t,
\qquad
D_k=\frac{e_k-e_{k-1}}{\Delta t},
```

```math
u_k=K_pe_k+K_iI_k+K_dD_k.
```

The target is three rotary coordinates at `90°` and the prismatic coordinate at
`40 mm`. Gains are specified per coordinate in `part1_pid.m`.

## Running the simulations

In MATLAB, change to this directory and run:

```matlab
run('part1_not_pid.m')
run('part1_pid.m')
```

Both scripts present joint positions, joint velocities, and a 3D arm view. The
project notebook places these visual results next to the corresponding theory and
interpretation.

## Model scope

The stored parameters define the numerical study included in this repository.
Future hardware-oriented work can add a calibrated unit set, frame map, and
experimentally identified rigid-body parameters. The MATLAB and Python components
currently use their own model conventions and serve complementary simulation and
planning roles.
