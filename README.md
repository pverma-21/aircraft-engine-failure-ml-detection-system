# 🛩️ Aircraft Engine Failure Detection

## Remaining Useful Life (RUL) Prediction using NASA C-MAPSS Dataset

This project implements machine learning and deep learning models to predict the **Remaining Useful Life (RUL)** of aircraft turbofan engines before failure, using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

---

## 📋 Problem Statement

**Objective:** Predict the number of operational cycles an aircraft engine will continue to operate before failure, using multivariate time series sensor data.

**Why it matters:**
- Enables **predictive maintenance** instead of reactive repairs
- Reduces unscheduled downtime and **AOG (Aircraft on Ground)** events
- Optimizes maintenance scheduling and spare parts inventory
- Improves **flight safety** through early warning systems

---

## 📊 Dataset Overview

The NASA C-MAPSS dataset simulates degradation of aircraft turbofan engines with the following characteristics:

| Dataset | Train Engines | Test Engines | Operating Conditions | Fault Modes |
|---------|---------------|--------------|---------------------|-------------|
| **FD001** | 100 | 100 | 1 (Sea Level) | 1 (HPC Degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC Degradation) |
| FD003 | 100 | 100 | 1 (Sea Level) | 2 (HPC + Fan Degradation) |
| FD004 | 248 | 249 | 6 | 2 (HPC + Fan Degradation) |

### Data Structure (26 columns)
- **Column 1:** Engine unit number
- **Column 2:** Time (operational cycles)
- **Columns 3-5:** Operational settings (altitude, Mach, TRA)
- **Columns 6-26:** 21 sensor measurements

### Sensor Measurements
| Sensor | Description | Sensor | Description |
|--------|-------------|--------|-------------|
| T2 | Fan inlet temperature | P30 | HPC outlet pressure |
| T24 | LPC outlet temperature | Nf | Physical fan speed |
| T30 | HPC outlet temperature | Nc | Physical core speed |
| T48 | HPT outlet temperature (EGT) | phi | Fuel flow ratio |
| T50 | LPT outlet temperature | BPR | Bypass ratio |
| P2 | Fan inlet pressure | htBleed | Bleed enthalpy |
| P15 | Bypass duct pressure | W31/W32 | Coolant bleed |

---

## 🗂️ Project Structure

```
aircraft-engine-failure-detection/
├── 01_data_exploration.ipynb    # EDA and visualization
├── 02_rul_prediction_model.ipynb # ML/DL model training
├── README.md                     # This file
└── requirements.txt              # Dependencies (optional)
```

---

## 📓 Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Load and inspect all FD001-FD004 datasets
- Compute RUL for training data
- Analyze sensor variability and identify low-variance sensors
- Visualize degradation patterns over engine lifecycle
- Correlation analysis between sensors and RUL
- Compare dataset characteristics

### 2. RUL Prediction Models (`02_rul_prediction_model.ipynb`)
Implements and compares 4 different approaches:

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | ML Ensemble | Baseline tree-based regressor |
| **XGBoost** | Gradient Boosting | Advanced ensemble with regularization |
| **LSTM** | Deep Learning | Sequence modeling with memory cells |
| **1D CNN** | Deep Learning | Convolutional feature extraction |

---

## 🔧 Feature Engineering

1. **Rolling Statistics:** 5-cycle rolling mean and standard deviation for each sensor
2. **Low-Variance Sensor Removal:** T2, P2, P15, Nf_dmd, PCNfR_dmd, farB
3. **RUL Capping:** Maximum RUL capped at 125 cycles (piecewise linear assumption)
4. **Standardization:** StandardScaler normalization

---

## 📈 Evaluation Metrics

### PHM Scoring Function (Primary Metric)
The PHM'08 competition scoring function penalizes **late predictions more heavily** than early ones:

$$s = \sum_{i=1}^{n} \begin{cases} e^{-d/13} - 1 & \text{if } d < 0 \text{ (early)} \\ e^{d/10} - 1 & \text{if } d \geq 0 \text{ (late)} \end{cases}$$

where $d = \text{Predicted RUL} - \text{True RUL}$

### Additional Metrics
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score**

---

## 🚀 Getting Started

### Prerequisites
```bash
# Core dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Optional: Advanced ML
pip install xgboost

# Optional: Deep Learning
pip install tensorflow
```

### Running the Notebooks

1. **Clone/Download** the project
2. **Download the dataset** from [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
3. Place data files in: `../DataSets/aircraft-engine-failure-data/`
4. Open notebooks in Jupyter/VS Code and run cells sequentially

### Data Path Configuration
Update the `DATA_PATH` variable in notebooks if your data is in a different location:
```python
DATA_PATH = Path(r"path/to/your/aircraft-engine-failure-data")
```

---

## 📊 Expected Results (FD001)

| Model | RMSE | MAE | PHM Score |
|-------|------|-----|-----------|
| Random Forest | ~18-22 | ~14-17 | ~300-500 |
| XGBoost | ~17-21 | ~13-16 | ~280-450 |
| LSTM | ~15-20 | ~12-15 | ~250-400 |
| 1D CNN | ~16-20 | ~12-16 | ~260-420 |

*Results may vary based on random seeds and hyperparameters.*

---

## 🔬 Technical Details

### LSTM Architecture
```
LSTM(64, return_sequences=True)
├── Dropout(0.2)
├── BatchNormalization
├── LSTM(32)
├── Dropout(0.2)
├── BatchNormalization
├── Dense(32, relu)
├── Dropout(0.2)
├── Dense(16, relu)
└── Dense(1)
```

### 1D CNN Architecture
```
Conv1D(64, kernel=5, relu)
├── BatchNorm → MaxPool(2) → Dropout(0.2)
├── Conv1D(128, kernel=3, relu)
├── BatchNorm → MaxPool(2) → Dropout(0.2)
├── Conv1D(64, kernel=3, relu)
├── BatchNorm → Flatten
├── Dense(64, relu) → Dropout(0.3)
├── Dense(32, relu)
└── Dense(1)
```

---

## 📚 References

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). **Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation**. International Conference on Prognostics and Health Management (PHM08).

2. Frederick, D., DeCastro, J., & Litt, J. (2007). **User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)**. NASA Technical Manual TM2007-215026.

3. NASA Prognostics Center of Excellence Data Repository:  
   https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

---

## 🚀 Future Improvements

- [ ] Bidirectional LSTM / Attention mechanisms
- [ ] Transformer-based models
- [ ] Ensemble of multiple models
- [ ] Hyperparameter tuning with Optuna/GridSearch
- [ ] Physics-informed neural networks (PINN)
- [ ] Extend to FD002, FD003, FD004 datasets
- [ ] Deploy as REST API for real-time predictions

---

## 📄 License

This project is for educational and research purposes. The NASA C-MAPSS dataset is publicly available for non-commercial use.

---

## 👤 Author

**Prashant Verma**

---

*Built with ❤️ for Predictive Maintenance & Prognostics*

