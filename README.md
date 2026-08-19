# Predicting Stellar Class

An exploration of multiclass stellar-object classification for Kaggle's
[Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6/overview)
competition (Playground Series, Season 6 Episode 6).

The goal is to classify each observation as a **GALAXY**, **STAR**, or **QSO**
from tabular astronomical measurements. Submissions are evaluated using
**balanced accuracy**, so performance on each class matters even when the class
distribution is uneven.

## Approach

This repository develops an XGBoost solution iteratively:

1. Explore the data, feature distributions, and class balance.
2. Train a baseline multiclass XGBoost model.
3. Add astronomy-inspired and statistical features, including magnitude
   summaries, color indices, cyclical sky-coordinate transforms, and
   redshift interactions.
4. Improve validation with stratified K-fold cross-validation, class-balanced
   sample weights, and multiple random seeds.

## Repository structure

```text
.
├── EDA.ipynb
└── Models/
    └── XGBoost/
        ├── XGBoost_v1.ipynb
        ├── XGBoost_v2.ipynb
        └── XGBoost_v3.ipynb
```

- `EDA.ipynb` downloads the competition data and explores numerical and
  categorical features.
- `XGBoost_v1.ipynb` establishes the baseline model and submission workflow.
- `XGBoost_v2.ipynb` expands the feature set.
- `XGBoost_v3.ipynb` adds stratified cross-validation, balanced sample weights,
  multiple seeds, and decision-threshold optimization.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/jona-nakai/Predicting-Stellar-Class.git
cd Predicting-Stellar-Class
```

### 2. Create an environment and install dependencies

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter kagglehub matplotlib numpy pandas python-dotenv scipy scikit-learn seaborn torch xgboost
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Authenticate with Kaggle

The notebooks use `kagglehub` to download the competition data. Configure your
Kaggle credentials and accept the competition rules on the
[competition page](https://www.kaggle.com/competitions/playground-series-s6e6/overview)
before running them.

Keep credentials out of Git. Local data and `.env` files are ignored by this
repository.

### 4. Run the notebooks

```bash
jupyter lab
```

Start with `EDA.ipynb`, then work through the XGBoost notebooks in version
order. Generated submission files can be uploaded on Kaggle's
[submission page](https://www.kaggle.com/competitions/playground-series-s6e6/submit).

## Competition

- **Host:** Kaggle
- **Series:** Playground Series, Season 6 Episode 6
- **Task:** Multiclass tabular classification
- **Target classes:** GALAXY, STAR, and QSO
- **Evaluation metric:** Balanced accuracy
- **Competition link:** [Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6/overview)

Competition data is not included in this repository. Use of the data is subject
to Kaggle's competition rules.
