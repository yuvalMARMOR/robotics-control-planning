# Robotics Control and Learning-Guided Motion Planning

Simulation, control, and motion planning for a four-degree-of-freedom (4-DOF) robotic arm. The repository combines a MATLAB study of simplified open-loop dynamics and PID position control with a Python implementation of GMM-guided trajectory sampling and MPC-style receding-horizon selection.

> **Start here:** open the [technical Jupyter notebook](notebooks/robotics_control_planning.ipynb) for the complete **theory -> source code -> visual result -> interpretation** walkthrough, including derivations, implementation cross-references, result provenance, and verified limitations.

<p align="center">
  <a href="notebooks/robotics_control_planning.ipynb"><strong>Open the technical notebook</strong></a>
  &nbsp;·&nbsp;
  <a href="#mathematical-foundations"><strong>Mathematical foundations</strong></a>
  &nbsp;·&nbsp;
  <a href="#running-the-project"><strong>Run the project</strong></a>
</p>

<p align="center">
  <img src="assets/images/system-architecture.png" alt="System architecture for the MATLAB control and Python motion-planning workflows" width="100%">
</p>

> [!NOTE]
> The figures below are preserved historical simulation outputs. The tracked source and the supplied figures are not fully version-aligned; the notebook labels verified source behavior, historical claims, reconstructed diagnostics, and unresolved details separately.

## At a glance

| Area | MATLAB control study | Python planning study |
|---|---|---|
| Robot representation | Simplified 3R+1P dynamic model | 4-DOF kinematic model |
| Core method | Open-loop integration and joint-space PID | Sampled, receding-horizon trajectory selection |
| Learned component | None | Gaussian Mixture Model fitted to synthetic feasible trajectories |
| Environment | Gravity-related model terms | Three spherical workspace obstacles |
| Outputs | Joint positions, velocities, and 3D arm views | Task-space/configuration-space plots, sampling plots, GIFs, and summary figures |
| Primary implementation | [`src/matlab/`](src/matlab/) | [`src/python/part2_control.py`](src/python/part2_control.py) |

The Python implementation is accurately described as **GMM-guided trajectory sampling with an MPC-style cost function**. It is inspired by learned-sampling research, but it does **not** implement a Conditional Variational Autoencoder (CVAE) or a normalizing-flow network.

## System model

The arm is represented by three revolute coordinates and one prismatic coordinate:

```math
q =
\begin{bmatrix}
\theta_1 & \theta_2 & \theta_3 & d_4
\end{bmatrix}^{T}.
```

<p align="center">
  <img src="assets/images/robot-arm-kinematic-structure.jpg" alt="Reported kinematic structure and coordinate frames of the robotic arm" width="440">
</p>

The MATLAB and Python programs use different frame and parameter conventions. The schematic therefore communicates the reported mechanism conceptually; it is not an exact frame specification for both implementations.

## Mathematical foundations

### Lagrangian dynamics

The reported dynamics begin with kinetic and potential energy:

```math
L(q,\dot q) = T(q,\dot q) - U(q),
```

```math
T_i = \frac{1}{2}m_i v_i^2 + \frac{1}{2}I_i\dot\theta_i^2,
\qquad
U = \sum_{i=1}^{5}m_i g h_i(q).
```

Applying the Euler-Lagrange equation,

```math
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot q_i}\right)
\quad - \frac{\partial L}{\partial q_i}
= \tau_i,
```

gives the conventional manipulator form

```math
M(q)\ddot q + C(q,\dot q)\dot q + G(q) = \tau.
```

The MATLAB source implements a project-specific simplification using a constant diagonal inertia matrix and a configuration-dependent vector named `coriolisMat`. The notebook documents where this implementation differs from the conventional decomposition above.

### PID position control

For reference $r(t)$ and measured output $y(t)$, the tracking error is

```math
e(t) = r(t) - y(t),
```

and the ideal continuous-time PID law is

```math
u(t) = K_p e(t)
\quad + K_i\int_0^t e(\tau)\,d\tau
\quad + K_d\frac{de(t)}{dt}.
```

The reported target is

```math
q_d =
\begin{bmatrix}
90^\circ & 90^\circ & 90^\circ & 40\,\mathrm{mm}
\end{bmatrix}^{T}.
```

The controller study uses manually selected gains for every coordinate. Gain mapping, state feedback, integration, and prismatic-axis issues found in the tracked script are explicitly analyzed in the notebook.

### GMM-guided trajectory sampling

For each planning scenario, the Python program generates straight, via-point, random, and obstacle-aware trajectory candidates. Feasible trajectories are flattened and used to fit a Gaussian mixture:

```math
p(x) = \sum_{k=1}^{K}\pi_k\,
\mathcal{N}(x\mid\mu_k,\Sigma_k),
\qquad K \le 3.
```

New trajectories are sampled from the fitted model and combined with heuristic exploration samples. Each candidate is checked against joint limits and the implemented point-based collision test.

### MPC-style candidate objective

The implemented scoring function combines configuration error, task-space error, control effort, obstacle proximity, smoothness, and a terminal penalty. Define the stage cost as

```math
\ell_t =
w_t e_{q,t}^{T}Q_{\mathrm{pos}}e_{q,t}
\quad + 50w_t\lVert p(q_t)-p(q_g)\rVert_2^2
\quad + \Delta q_t^{T}R\Delta q_t
\quad + J_{\mathrm{obs},t}
\quad + 0.1\left\lVert\frac{\Delta q_t}{\Delta t}\right\rVert_2^2.
```

The complete candidate cost is

```math
J = \sum_{t=0}^{N-1}\ell_t
\quad + 100\lVert q_{N-1}-q_g\rVert_2^2.
```

The controller selects the lowest-cost feasible trajectory and applies only its first action:

```math
\dot q_k = \mathrm{clip}\!\left(
2\frac{q_1^{\star}-q_0^{\star}}{\Delta t},-1,1
\right),
\qquad
q_{k+1}=q_k+\dot q_k\Delta t.
```

This is receding-horizon selection over sampled kinematic trajectories, not torque-level nonlinear MPC.

## Visual results

### Open-loop dynamics and PID response

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/images/open-loop-position-response.png" alt="Preserved open-loop joint-position response"><br>
      <strong>Open-loop joint response</strong>
    </td>
    <td width="50%" align="center">
      <img src="assets/images/pid-position-response.png" alt="Preserved PID-controlled joint-position response"><br>
      <strong>PID-controlled joint response</strong>
    </td>
  </tr>
</table>

The open-loop curves show increasing coordinates under fixed numerical inputs. The PID figure shows motion toward the reported targets, followed by trailing zeros caused by early loop termination and untouched preallocated samples. These are preserved results rather than newly reproduced benchmarks.

### Point-to-point motion planning

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/images/planning-task-space-point-to-point.png" alt="Point-to-point end-effector path in task space"><br>
      <strong>Task-space trajectory</strong>
    </td>
    <td width="50%" align="center">
      <img src="assets/images/planning-configuration-space-point-to-point.png" alt="Point-to-point trajectory in configuration space"><br>
      <strong>Configuration-space response</strong>
    </td>
  </tr>
</table>

The preserved point-to-point figures show a smooth end-effector curve and convergence toward the displayed joint targets, qualitatively reflecting the goal-tracking and smoothness terms in the candidate cost.

### Complex maneuvering

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/images/planning-task-space-complex-maneuvering.png" alt="Complex end-effector maneuver in task space"><br>
      <strong>Task-space maneuver</strong>
    </td>
    <td width="50%" align="center">
      <img src="assets/images/planning-configuration-space-complex-maneuvering.png" alt="Complex maneuver in configuration space"><br>
      <strong>Configuration-space response</strong>
    </td>
  </tr>
</table>

The more complex response includes non-monotonic corrections, especially in the prismatic coordinate. Exact historical run settings remain unresolved because the supplied figures and tracked scenario definitions are not fully version-aligned.

## Repository structure

```text
.
├── assets/images/                   # Curated historical figures and architecture diagram
├── notebooks/
│   └── robotics_control_planning.ipynb
├── src/
│   ├── matlab/
│   │   ├── part1_not_pid.m
│   │   └── part1_pid.m
│   └── python/
│       └── part2_control.py
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

### MATLAB

- MATLAB with core plotting and matrix operations.
- No project-specific toolbox dependency is evident from the source.

The historical MATLAB release is not recorded, so no tested-version claim is made.

### Python

- Python 3.9 or newer is recommended.
- Direct dependencies are listed in [`requirements.txt`](requirements.txt).

Create an isolated environment:

```bash
python -m venv .venv
```

Activate it and install the dependencies.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running the project

### MATLAB simulations

Change to `src/matlab/` in MATLAB and run:

```matlab
run('part1_not_pid.m')  % Open-loop simulation
run('part1_pid.m')      % PID-controlled simulation
```

The scripts display figures interactively but do not save result files.

### Python planning study

From the repository root:

```bash
python src/python/part2_control.py
```

The script creates `mpc_motion_planning_output/` relative to the current working directory. Depending on successful completion, it may contain configuration-space plots, task-space plots, sampling plots, GIF animations, and a summary figure.

## Result provenance

| Study | Available evidence | Status |
|---|---|---|
| Open-loop dynamics | Position, velocity, and arm-configuration figures | Preserved historical output |
| PID control | Position, velocity, arm configuration, and reported terminal values | Preserved historical output; not independently reproduced |
| Point-to-point planning | Task-space and configuration-space figures | Preserved historical output |
| Complex maneuvering | Task-space and configuration-space figures | Preserved historical output |
| Link-collision comparison | Endpoint-versus-segment diagnostic table in the notebook | Reconstructed diagnostic |
| CVAE performance | Referenced research theory | Not produced by this implementation |

No raw historical result arrays, dependency lockfile, run manifest, or classical-planner baseline accompanied the figures.

## Known limitations

- The supplied figures appear to originate from an earlier source revision in several places.
- The MATLAB model mixes unit systems and contains verified state-update, gain-mapping, and prismatic-axis issues.
- Python collision checking tests joint positions rather than complete link segments.
- The Python planner uses a GMM, not the CVAE described by the referenced research method.
- Physical dynamics are not integrated into the Python planning loop.
- There is no automated test suite or independently reproduced performance baseline.
- Generated PNG and GIF outputs are ignored by default; only curated documentation assets are tracked.

## License

Released under the [MIT License](LICENSE).
