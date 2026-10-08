# PEST Calibration Diagnostic App

A multi-page Dash application for diagnosing PEST / PEST++ parameter optimisation runs.

## Pages

| Page | Description |
|------|-------------|
| Objective Function | Φ convergence, Marquardt λ, per-group breakdown |
| Residuals | Obs vs sim, weighted residuals, Q-Q plot |
| Contribution Matrix | C[i,j] = −rᵢ·J[i,j]·Δθⱼ heatmap |
| Sensitivity | CSS ranking, OSS heatmap, parameter dominance |
| Identifiability | SVD scores, null-space, correlation matrix |
| Regularisation | Φ trade-off curve, regularisation weight factor |
| Convergence | Parameter change trajectories, model call efficiency |
| Data Worth | Leverage, influence, predictive worth |
| Uncertainty | FOSM prior/posterior std, prediction CIs |

## Quick start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8050
```

## Using real PEST data

Edit `app.py` and uncomment:

```python
from data.pest_reader import read_pest_run
PEST_DATA = read_pest_run("/path/to/your/pest_run_basename")
```

The parser reads: `.rec`, `.rei`, `.par`, `.sen`, `.jco`, `.jcb`, `.cov`

## File structure

```
pest_app/
├── app.py               # App shell + routing
├── requirements.txt
├── README.md
├── assets/
│   └── custom.css
├── data/
│   └── pest_reader.py   # PEST file parser
├── components/
│   └── sidebar.py
└── pages/
    ├── objective_function.py
    ├── residuals.py
    ├── contribution.py
    ├── sensitivity.py
    ├── identifiability.py
    ├── regularisation.py
    ├── convergence.py
    ├── data_worth.py
    └── uncertainty.py
```
