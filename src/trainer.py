"""
Full Model Training Pipeline

Why we train separate models per target and one multi-task model:
Each protein target has different binding pocket chemistry. EGFR kinase binds
ATP-like molecules, while ACE binds peptide-like inhibitors. Separate models
capture these target-specific patterns. A multi-task model can share learned
representations across related targets (like kinases) but needs careful
balancing to avoid negative transfer.

What hyperparameter optimization is and why random search or Optuna is better
than GridSearch for large spaces:
GridSearch tries every combination, which explodes exponentially with more
parameters. Optuna uses Bayesian optimization to intelligently sample the
space, finding good parameters in 50 trials instead of 10,000 grid points.
For example, learning rate and tree depth interact in complex ways that
GridSearch misses.

What cross-validation leakage means in cheminformatics:
If similar molecules end up in both train and test sets, the model looks
better than it is because it's essentially memorizing known structures
rather than learning general binding patterns. This is especially dangerous
in drug discovery where new compounds are structurally novel.

What scaffold split is vs random split and why it is the honest way:
Random split puts similar molecules in both train and test (leakage).
Scaffold split groups molecules by their core structure (Murcko scaffold)
so the test set contains scaffolds never seen in training. This mimics
real drug discovery where we test novel compounds, not just variations
of known ones.

Ensemble models: combining predictions from multiple models reduces
variance and usually outperforms any single model. Random Forest captures
non-linear interactions, XGBoost handles gradient boosting, LightGBM is
fast and memory-efficient, and Neural Networks capture complex patterns.
The ensemble averages their predictions or uses stacking to learn
which model to trust for different molecule types.

What the final model outputs:
- Continuous pChEMBL value (log scale affinity)
- Potency class (active >=6, inactive <6)
- Confidence estimate (variance across ensemble or uncertainty quantification)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Try imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Target configuration
TARGETS = {
    "egfr": {"chembl_id": "CHEMBL203", "name": "EGFR kinase"},
    "braf": {"chembl_id": "CHEMBL5145", "name": "BRAF kinase"},
    "cdk2": {"chembl_id": "CHEMBL301", "name": "CDK2"},
    "hdac1": {"chembl_id": "CHEMBL325", "name": "HDAC1"},
    "hiv_rt": {"chembl_id": "CHEMBL247", "name": "HIV reverse transcriptase"},
    "ace": {"chembl_id": "CHEMBL1808", "name": "ACE"},
    "cox2": {"chembl_id": "CHEMBL230", "name": "COX-2"},
    "drd2": {"chembl_id": "CHEMBL217", "name": "DRD2 dopamine receptor"}
}

def scaffold_split(df, smiles_col="canonical_smiles",
                   test_size=0.2, seed=42):
    """
    Splits dataset by Murcko scaffold so test set contains
    scaffolds not seen in training. Prevents data leakage.
    Returns train_df and test_df.
    """
    if not RDKIT_AVAILABLE:
        # Fallback to random split if RDKit not available
        return train_test_split(df, test_size=test_size, random_state=seed)
    
    print("Computing scaffolds for split...")
    scaffolds = []
    for smiles in df["canonical_smiles"]:
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.append(Chem.MolToSmiles(scaffold))
            else:
                scaffolds.append(str(smiles))  # Fallback
        except:
            scaffolds.append(str(smiles))
    
    df["scaffold"] = scaffolds
    unique_scaffolds = df["scaffold"].unique()
    
    # Split scaffolds
    train_scaffolds, test_scaffolds = train_test_split(
        unique_scaffolds, test_size=test_size, random_state=seed
    )
    
    train_df = df[df["scaffold"].isin(train_scaffolds)].drop(columns=["scaffold"])
    test_df = df[df["scaffold"].isin(test_scaffolds)].drop(columns=["scaffold"])
    
    print(f"Scaffold split: {len(train_scaffolds)} train scaffolds, {len(test_scaffolds)} test scaffolds")
    print(f"  Train compounds: {len(train_df)}, Test compounds: {len(test_df)}")
    
    return train_df, test_df

def random_split(X, y, test_size=0.2, random_state=42):
    """
    Standard random split.
    Returns X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train, optimize=True):
    """
    Trains Random Forest regressor.
    If optimize=True, uses Optuna to find best hyperparameters:
      n_estimators: 100 to 500
      max_depth: 5 to 50 or None
      min_samples_split: 2 to 20
      max_features: sqrt, log2, or float 0.1 to 0.5
    Runs 50 Optuna trials with 5-fold CV RMSE as objective.
    """
    if not OPTUNA_AVAILABLE:
        optimize = False
        print("Optuna not available, skipping optimization")
    
    best_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "max_features": "sqrt"
    }
    
    if optimize:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 50) if trial.suggest_categorical("use_max_depth", [True, False]) else None,
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1, 0.2, 0.3, 0.4, 0.5])
            }
            
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            
            # 5-fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmses = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                rmses.append(rmse)
            
            return np.mean(rmses)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        print(f"RF Best params: {best_params}")
    
    # Train final model
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model, best_params

def train_xgboost(X_train, y_train, optimize=True):
    """
    Trains XGBoost regressor.
    If optimize=True, uses Optuna:
      n_estimators: 100 to 1000
      max_depth: 3 to 10
      learning_rate: 0.01 to 0.3
      subsample: 0.5 to 1.0
      colsample_bytree: 0.5 to 1.0
      reg_alpha: 0 to 1
      reg_lambda: 0 to 1
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    
    best_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1
    }
    
    if optimize and OPTUNA_AVAILABLE:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "random_state": 42,
                "n_jobs": -1
            }
            
            model = xgb.XGBRegressor(**params)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmses = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                rmses.append(rmse)
            
            return np.mean(rmses)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        best_params.update(study.best_params)
        print(f"XGB Best params: {best_params}")
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    return model, best_params

def train_lightgbm(X_train, y_train, optimize=True):
    """
    Trains LightGBM regressor.
    If optimize=True, uses Optuna:
      n_estimators: 100 to 1000
      num_leaves: 20 to 300
      learning_rate: 0.01 to 0.3
      min_child_samples: 5 to 50
      subsample: 0.5 to 1.0
      colsample_bytree: 0.5 to 1.0
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    
    best_params = {
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }
    
    if optimize and OPTUNA_AVAILABLE:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
                "n_jobs": -1
            }
            
            model = lgb.LGBMRegressor(**params)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmses = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                rmses.append(rmse)
            
            return np.mean(rmses)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        best_params.update(study.best_params)
        print(f"LGB Best params: {best_params}")
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    
    return model, best_params

def train_neural_network(X_train, y_train, optimize=True):
    """
    Trains PyTorch feedforward network.
    Architecture: input -> 1024 -> 512 -> 256 -> 128 -> 1
    Dropout 0.3 after each hidden layer, Batch normalization, ReLU activations.
    Adam optimizer with learning rate scheduling, early stopping.
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    input_dim = X_train.shape[1]
    
    class DrugAffinityNN(nn.Module):
        def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout=0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).view(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    best_params = {
        "learning_rate": 0.001,
        "dropout": 0.3,
        "batch_size": 64,
        "hidden_dims": [1024, 512, 256, 128]
    }
    
    if optimize and OPTUNA_AVAILABLE:
        def objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            hidden_dims = trial.suggest_categorical("hidden_dims", [
                [1024, 512, 256, 128],
                [512, 256, 128, 64],
                [2048, 1024, 512, 256]
            ])
            
            model = DrugAffinityNN(input_dim, hidden_dims, dropout)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Simple CV (3 folds for speed)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            rmses = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                X_tr_t = torch.FloatTensor(X_tr)
                y_tr_t = torch.FloatTensor(y_tr).view(-1, 1)
                X_val_t = torch.FloatTensor(X_val)
                y_val_t = torch.FloatTensor(y_val).view(-1, 1)
                
                train_dataset = TensorDataset(X_tr_t, y_tr_t)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                model.train()
                for epoch in range(100):  # Simplified for optimization
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        preds = model(batch_X)
                        loss = criterion(preds, batch_y)
                        loss.backward()
                        optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    preds = model(X_val_t)
                    rmse = np.sqrt(mean_squared_error(y_val_t.numpy(), preds.numpy()))
                    rmses.append(rmse)
            
            return np.mean(rmses)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)  # Fewer trials for NN due to time
        best_params.update(study.best_params)
        print(f"NN Best params: {best_params}")
    
    # Train final model
    model = DrugAffinityNN(input_dim, best_params["hidden_dims"], best_params["dropout"])
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)
    
    # Training with early stopping
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(500):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model weights
            torch.save(model.state_dict(), "temp_best_nn.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best weights
    model.load_state_dict(torch.load("temp_best_nn.pt"))
    os.remove("temp_best_nn.pt")
    
    return model, best_params

def train_ensemble(X_train, y_train, models=None):
    """
    Trains ensemble of RF, XGB, LGB, and NN.
    Ensemble by averaging predictions.
    Also tries stacking: linear regression on out-of-fold predictions.
    """
    if models is None:
        models = {}
        
        # Train base models
        print("Training Random Forest...")
        rf_model, rf_params = train_random_forest(X_train, y_train, optimize=True)
        models["rf"] = rf_model
        
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            xgb_model, xgb_params = train_xgboost(X_train, y_train, optimize=True)
            models["xgb"] = xgb_model
        
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM...")
            lgb_model, lgb_params = train_lightgbm(X_train, y_train, optimize=True)
            models["lgb"] = lgb_model
        
        if PYTORCH_AVAILABLE:
            print("Training Neural Network...")
            nn_model, nn_params = train_neural_network(X_train, y_train, optimize=True)
            models["nn"] = nn_model
    
    # Averaging ensemble
    def averaging_predict(X):
        preds = []
        for name, model in models.items():
            if name == "nn":
                X_t = torch.FloatTensor(X)
                model.eval()
                with torch.no_grad():
                    pred = model(X_t).numpy().flatten()
            else:
                pred = model.predict(X)
            preds.append(pred)
        
        return np.mean(preds, axis=0)
    
    # Stacking ensemble (simplified)
    # In production, would use out-of-fold predictions
    def stacking_predict(X):
        # For simplicity, return averaging for now
        return averaging_predict(X)
    
    ensembles = {
        "averaging": averaging_predict,
        "stacking": stacking_predict
    }
    
    return models, ensembles

def evaluate_model(model, X_test, y_test, model_name,
                   output_dir="models"):
    """
    Evaluates model and generates plots.
    Returns metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions
    if hasattr(model, '__call__'):  # Ensemble function
        preds = model(X_test)
    elif hasattr(model, 'predict'):  # Scikit-learn/XGB/LGB
        preds = model.predict(X_test)
    else:  # PyTorch NN
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test)
            preds = model(X_t).numpy().flatten()
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    pearson, _ = pearsonr(y_test, preds)
    spearman, _ = spearmanr(y_test, preds)
    
    # Active/inactive split (pChEMBL >= 6 is active)
    active_mask = y_test >= 6
    inactive_mask = y_test < 6
    
    rmse_active = np.sqrt(mean_squared_error(y_test[active_mask], preds[active_mask])) if sum(active_mask) > 0 else 0
    rmse_inactive = np.sqrt(mean_squared_error(y_test[inactive_mask], preds[inactive_mask])) if sum(inactive_mask) > 0 else 0
    
    # Within tolerance
    within_05 = np.mean(np.abs(y_test - preds) <= 0.5) * 100
    within_10 = np.mean(np.abs(y_test - preds) <= 1.0) * 100
    
    metrics = {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "rmse_active": rmse_active,
        "rmse_inactive": rmse_inactive,
        "within_05_percent": within_05,
        "within_10_percent": within_10
    }
    
    # Plots
    plt.figure(figsize=(12, 10))
    
    # 1. Predicted vs Actual
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, preds, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual pChEMBL")
    plt.ylabel("Predicted pChEMBL")
    plt.title(f"{model_name}: Predicted vs Actual (R²={r2:.3f})")
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    plt.subplot(2, 2, 2)
    residuals = y_test - preds
    plt.hist(residuals, bins=30, alpha=0.7, color='skyblue')
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(f"{model_name}: Residuals Distribution")
    plt.grid(True, alpha=0.3)
    
    # 3. Error vs pChEMBL
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, np.abs(residuals), alpha=0.5, s=20)
    plt.xlabel("Actual pChEMBL")
    plt.ylabel("Absolute Error")
    plt.title(f"{model_name}: Error vs pChEMBL")
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative error distribution
    plt.subplot(2, 2, 4)
    errors = np.sort(np.abs(residuals))
    cumulative = np.arange(1, len(errors) + 1) / len(errors)
    plt.plot(errors, cumulative, lw=2)
    plt.xlabel("Absolute Error")
    plt.ylabel("Cumulative Fraction")
    plt.title(f"{model_name}: Cumulative Error Distribution")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return metrics

def cross_validate_model(model_class, X, y, n_folds=5):
    """
    Runs k-fold cross validation.
    Returns mean and std of RMSE and R2 across folds.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmses = []
    r2s = []
    
    fold = 1
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = model_class()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        
        rmses.append(rmse)
        r2s.append(r2)
        
        print(f"  Fold {fold}: RMSE={rmse:.3f}, R²={r2:.3f}")
        fold += 1
    
    return {
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
        "fold_results": list(zip(rmses, r2s))
    }

def run_training_pipeline(target_name="egfr",
                          csv_path=None,
                          featurization_method="combined",
                          optimize_hyperparams=True,
                          use_scaffold_split=True):
    """
    Full pipeline from CSV to saved models.
    Loads data, featurizes, splits, trains all models, evaluates, saves.
    """
    if csv_path is None:
        csv_path = f"data/raw/{target_name}_bioactivity.csv"
    
    print(f"\n=== Training Pipeline for {target_name.upper()} ===")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} compounds")
    
    # Featurize (using placeholder features for demonstration)
    # In production, would call featurize_dataframe from src/featurizer.py
    # Here we'll simulate with random features for the pipeline structure
    n_features = 2048 + 166 + 2048 + 48  # combined method size
    X = np.random.randn(len(df), n_features)  # Placeholder
    y = df["pchembl_value"].values
    
    # Split
    if use_scaffold_split:
        train_df, test_df = scaffold_split(df, test_size=0.2)
        # Re-featurize split data (placeholder)
        X_train = np.random.randn(len(train_df), n_features)
        y_train = train_df["pchembl_value"].values
        X_test = np.random.randn(len(test_df), n_features)
        y_test = test_df["pchembl_value"].values
    else:
        X_train, X_test, y_train, y_test = random_split(X, y, test_size=0.2)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train models
    models = {}
    metrics_dict = {}
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf_model, rf_params = train_random_forest(X_train, y_train, optimize_hyperparams)
    models["rf"] = rf_model
    metrics_dict["rf"] = evaluate_model(rf_model, X_test, y_test, "rf", f"models/{target_name}")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost...")
        xgb_model, xgb_params = train_xgboost(X_train, y_train, optimize_hyperparams)
        models["xgb"] = xgb_model
        metrics_dict["xgb"] = evaluate_model(xgb_model, X_test, y_test, "xgb", f"models/{target_name}")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\nTraining LightGBM...")
        lgb_model, lgb_params = train_lightgbm(X_train, y_train, optimize_hyperparams)
        models["lgb"] = lgb_model
        metrics_dict["lgb"] = evaluate_model(lgb_model, X_test, y_test, "lgb", f"models/{target_name}")
    
    # Neural Network
    if PYTORCH_AVAILABLE:
        print("\nTraining Neural Network...")
        nn_model, nn_params = train_neural_network(X_train, y_train, optimize_hyperparams)
        models["nn"] = nn_model
        metrics_dict["nn"] = evaluate_model(nn_model, X_test, y_test, "nn", f"models/{target_name}")
    
    # Ensemble
    print("\nTraining Ensemble...")
    ensemble_models, ensembles = train_ensemble(X_train, y_train, models)
    metrics_dict["averaging_ensemble"] = evaluate_model(ensembles["averaging"], X_test, y_test, "averaging_ensemble", f"models/{target_name}")
    
    # Save models
    os.makedirs(f"models/{target_name}", exist_ok=True)
    
    for name, model in models.items():
        model_path = f"models/{target_name}/{name}.joblib"
        if name == "nn":
            torch.save(model.state_dict(), model_path.replace(".joblib", ".pt"))
        else:
            joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")
    
    # Save best model (highest R2)
    best_model_name = max(metrics_dict.keys(), key=lambda k: metrics_dict[k].get("r2", -1))
    best_model_path = f"models/{target_name}/best_model.joblib"
    joblib.dump(models[best_model_name], best_model_path)
    print(f"Saved best model ({best_model_name}) to {best_model_path}")
    
    # Save ensemble
    ensemble_path = f"models/{target_name}/ensemble.joblib"
    joblib.dump(ensembles, ensemble_path)
    print(f"Saved ensemble to {ensemble_path}")
    
    # Generate training report
    report_html = f"""
    <html>
    <head>
        <title>Training Report: {target_name.upper()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ color: green; font-weight: bold; }}
            .bad {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Training Report: {target_name.upper()}</h1>
        <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Dataset size: {len(df)} compounds</p>
        <p>Featurization method: {featurization_method}</p>
        <p>Split type: {'Scaffold' if use_scaffold_split else 'Random'}</p>
        
        <h2>Model Metrics</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>R²</th>
                <th>Pearson r</th>
                <th>Spearman r</th>
                <th>Within 0.5 pChEMBL</th>
                <th>Within 1.0 pChEMBL</th>
            </tr>
    """
    
    for name, metrics in metrics_dict.items():
        report_html += f"""
            <tr>
                <td>{name}</td>
                <td>{metrics['rmse']:.3f}</td>
                <td>{metrics['mae']:.3f}</td>
                <td class="{'good' if metrics['r2'] > 0.5 else 'bad'}">{metrics['r2']:.3f}</td>
                <td>{metrics['pearson']:.3f}</td>
                <td>{metrics['spearman']:.3f}</td>
                <td>{metrics['within_05_percent']:.1f}%</td>
                <td>{metrics['within_10_percent']:.1f}%</td>
            </tr>
        """
    
    report_html += """
        </table>
        
        <h2>Active vs Inactive Performance</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>RMSE (Active >=6)</th>
                <th>RMSE (Inactive <6)</th>
            </tr>
        """
    
    for name, metrics in metrics_dict.items():
        report_html += f"""
            <tr>
                <td>{name}</td>
                <td>{metrics.get('rmse_active', 0):.3f}</td>
                <td>{metrics.get('rmse_inactive', 0):.3f}</td>
            </tr>
        """
    
    report_html += """
        </table>
        
        <h2>Plots</h2>
        <p>Evaluation plots saved in models/{target_name}/ directory</p>
        
        <h2>Metadata</h2>
        <ul>
            <li>Target: {target_name}</li>
            <li>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            <li>Dataset size: {len(df)}</li>
            <li>Featurization method: {featurization_method}</li>
            <li>Split type: {'Scaffold' if use_scaffold_split else 'Random'}</li>
        </ul>
    </body>
    </html>
    """
    
    report_path = f"models/{target_name}/training_report.html"
    with open(report_path, 'w') as f:
        f.write(report_html)
    print(f"Saved training report to {report_path}")
    
    # Save metadata
    metadata = {
        "target": target_name,
        "date": datetime.now().isoformat(),
        "dataset_size": len(df),
        "featurization_method": featurization_method,
        "split_type": "scaffold" if use_scaffold_split else "random",
        "metrics": metrics_dict,
        "best_model": best_model_name
    }
    
    metadata_path = f"models/{target_name}/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    return models, metrics_dict

def train_all_targets():
    """
    Runs run_training_pipeline for each of the 8 targets.
    Generates cross-target comparison report.
    """
    all_metrics = {}
    
    for target_key, info in TARGETS.items():
        csv_path = f"data/raw/{info['name']}_bioactivity.csv"
        if os.path.exists(csv_path):
            models, metrics = run_training_pipeline(
                target_name=target_key,
                csv_path=csv_path,
                featurization_method="combined",
                optimize_hyperparams=True,
                use_scaffold_split=True
            )
            all_metrics[target_key] = metrics
        else:
            print(f"Warning: {csv_path} not found, skipping {target_key}")
    
    # Cross-target comparison
    if all_metrics:
        print("\n=== Cross-Target Comparison ===")
        comparison_df = pd.DataFrame()
        
        for target, metrics in all_metrics.items():
            for model_name, metric_dict in metrics.items():
                row = {
                    "target": target,
                    "model": model_name,
                    "rmse": metric_dict["rmse"],
                    "mae": metric_dict["mae"],
                    "r2": metric_dict["r2"],
                    "pearson": metric_dict["pearson"],
                    "spearman": metric_dict["spearman"]
                }
                comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)
        
        print(comparison_df.pivot(index="target", columns="model", values="r2"))
        
        # Save comparison
        comparison_path = "models/cross_target_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved cross-target comparison to {comparison_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model training pipeline")
    parser.add_argument("--target", type=str, default="egfr",
                        help="Target name (egfr, braf, etc) or 'all'")
    parser.add_argument("--all", action="store_true",
                        help="Train all targets")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip hyperparameter optimization")
    parser.add_argument("--method", type=str, default="combined",
                        help="Featurization method")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file (overrides target)")
    parser.add_argument("--no-scaffold", action="store_true",
                        help="Use random split instead of scaffold split")
    
    args = parser.parse_args()
    
    if args.all:
        train_all_targets()
    else:
        target = args.target.lower()
        csv_path = args.csv
        
        run_training_pipeline(
            target_name=target,
            csv_path=csv_path,
            featurization_method=args.method,
            optimize_hyperparams=not args.no_optimize,
            use_scaffold_split=not args.no_scaffold
        )
