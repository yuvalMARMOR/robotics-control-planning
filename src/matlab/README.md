# MATLAB legacy dynamics and PID study

The MATLAB scripts are a separate, simplified historical study. They are not the
dynamic plant for the Python planner and they are not a validated rigid-body
model.

The pre-repair source is preserved by Git tag `legacy-pre-technical-repair`.
The tracked scripts correct only implementation defects whose intent is clear:

- state and velocity vectors are refreshed every step;
- generalized inputs are explicit four-element column vectors;
- the configuration-dependent gravity expression is no longer multiplied by a
  velocity row vector or described as a conventional Coriolis matrix;
- position integration uses the old velocity and a single `0.5*a*dt^2` term;
- the PID integral includes `dt`, and the documented `Kp`, `Ki`, `Kd` ordering is
  applied directly;
- the prismatic coordinate uses acceleration component four;
- time starts at zero and the simulation no longer plots untouched trailing
  preallocation after an overshoot-triggered break.

The following cannot be repaired uniquely from repository evidence:

- physical units for length, mass, inertia, generalized input, and prismatic
  displacement;
- the degree/radian convention required by a physical dynamic model;
- a validated `M(q)`, `C(q,qdot)`, or `G(q)` derivation;
- whether the fourth diagonal term should be `I5` or `m5`;
- equivalence to the independent Python kinematic frames.

Consequently, new MATLAB output must be described as a run of the corrected
legacy numerical study, not as experimental or physically validated robot
performance. A MATLAB/Octave runtime was not available in the repair environment,
so the scripts received static review only.
