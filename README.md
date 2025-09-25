# Robotics Control & Planning

Simulation and control of a **4-DOF robotic arm** as part of a robotics mini-project.  
The project is divided into two major parts:

1. **MATLAB (Part 1):**
   - Derivation of dynamics for a 3R+1P robotic arm.
   - Open-loop simulation (without PID).
   - Closed-loop simulation with a PID controller.

2. **Python (Part 2):**
   - Motion planning using **Model Predictive Control (MPC)**.
   - Learning-based sampling with **Normalizing Flow**.
   - Simulation of obstacle avoidance and trajectory tracking.

---

## 📂 Project Structure

```
.
├── docs/
│   ├── final_project_report_robotics.pdf        # Full academic report
│   └── installation_and_execution_guide.pdf     # How to install and run
│
├── src/
│   ├── matlab/
│   │   ├── part1_not_pid.m                      # Dynamics simulation without PID
│   │   └── part1_pid.m                          # Dynamics simulation with PID
│   │
│   └── python/
│       └── part2_control.py                     # MPC motion planning with Normalizing Flow
│
├── .gitignore
└── LICENSE
```

---

## ⚙️ Requirements

### MATLAB (Part 1)
- MATLAB R2020a or newer (tested on R2022b).  
- No additional toolboxes required beyond core MATLAB.  

### Python (Part 2)
- Python **3.7+**  
- Required libraries:  
  ```
  pip install numpy matplotlib scipy scikit-learn
  ```

---

## ▶️ How to Run

### Part 1 (MATLAB)
1. Open MATLAB.  
2. Navigate to `src/matlab/`.  
3. Run either:  
   - `part1_not_pid.m` → Open-loop simulation.  
   - `part1_pid.m` → Closed-loop simulation with PID.  

### Part 2 (Python)
1. Navigate to `src/python/`.  
2. Run:  
   ```
   python part2_control.py
   ```
3. Results (plots & animations) will be saved in `mpc_motion_planning_output/`.  

---

## 📑 Documentation

- [Project Report](docs/final_project_report_robotics.pdf)  
- [Installation Guide](docs/installation_and_execution_guide.pdf)
 

---

## 📜 License
This project is released under the **MIT License**.
