import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# ==========================================
# CoolQuant Pillar 5: Causal Inference
# Double ML vs. Penalized Regression Bias
# ==========================================

def generate_confounded_data(n_samples=5000, n_features=50, true_causal_effect=2.5):
    """
    Generates synthetic market data where a Trading Signal (T) 
    has a true causal effect on Returns (Y).
    However, T and Y are completely confounded by 50 other noisy variables (W).
    """
    np.random.seed(42)
    
    # W: 50 different market regimes, macro indicators, and noisy features
    W = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # The true generating process mapping the confounders to the Treatment (our Signal T)
    # T is heavily dependent on the first 5 variables in W.
    gamma = np.zeros(n_features)
    gamma[:5] = [1.5, -2.0, 3.0, -1.0, 0.5]
    T = W @ gamma + np.random.normal(0, 1, size=n_samples)
    
    # The true generating process for Returns (Y).
    # Y is driven by the true_causal_effect of T, plus the confounding effect of W.
    beta = np.random.uniform(-1, 1, size=n_features)
    Y = true_causal_effect * T + W @ beta + np.random.normal(0, 1, size=n_samples)
    
    return W, T, Y, true_causal_effect

def naive_lasso_estimate(W, T, Y):
    """
    Model 1: The Predictive "Data Scientist" Approach
    We throw all features (W and T) into a Lasso Regression.
    Because T is highly correlated with W, the L1 penalty will arbitrarily
    shrink the coefficient of T, resulting in Regularization Bias.
    """
    # Combine Signal (T) with all other features (W)
    X = np.column_stack((T, W))
    
    # Fit Cross-Validated Lasso
    model = LassoCV(cv=5, random_state=42)
    model.fit(X, Y)
    
    # The first coefficient belongs to T
    estimated_effect = model.coef_[0]
    return estimated_effect

def double_ml_estimate(W, T, Y):
    """
    Model 2: The Causal "Senior Quant" Approach (Double ML / FWL Theorem)
    We use ML strictly to model the nuisance parameters (the confounding effects of W).
    By residualizing both Y and T with respect to W, we isolate the pure variance in T,
    allowing us to recover the exact true causal effect via OLS on the residuals.
    """
    # Cross-fitting to prevent overfitting bias
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    t_residuals = np.zeros_like(T)
    y_residuals = np.zeros_like(Y)
    
    # Stage 1: Predict T from W, and Y from W using highly flexible ML (Random Forests)
    for train_idx, test_idx in kf.split(W):
        W_train, W_test = W[train_idx], W[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Nuisance Model 1: De-bias the Treatment
        model_t = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model_t.fit(W_train, T_train)
        t_pred = model_t.predict(W_test)
        t_residuals[test_idx] = T_test - t_pred
        
        # Nuisance Model 2: De-bias the Outcome
        model_y = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model_y.fit(W_train, Y_train)
        y_pred = model_y.predict(W_test)
        y_residuals[test_idx] = Y_test - y_pred

    # Stage 2: Regress the Y residuals on the T residuals (Frisch-Waugh-Lovell Theorem)
    final_model = LinearRegression()
    # Reshape for sklearn
    final_model.fit(t_residuals.reshape(-1, 1), y_residuals)
    
    estimated_effect = final_model.coef_[0]
    return estimated_effect


if __name__ == "__main__":
    print("==========================================================")
    print("CoolQuant Causal Inference: Double ML vs. Lasso")
    print("==========================================================")
    
    print("\n[1] Generating highly confounded market data...")
    W, T, Y, true_effect = generate_confounded_data()
    print(f"    -> Ground Truth Causal Effect of Signal on Return: {true_effect:.4f}")
    
    print("\n[2] Fitting Naive Predictive Model (Lasso CV)...")
    lasso_effect = naive_lasso_estimate(W, T, Y)
    lasso_error = abs(true_effect - lasso_effect)
    print(f"    -> Lasso Estimated Effect: {lasso_effect:.4f}")
    print(f"    -> Penalty Bias (Error):   {lasso_error:.4f}  <-- THE DANGER OF SHRINKAGE")
    
    print("\n[3] Fitting Causal Model (Double Machine Learning with Non-linear Nuisance Models)...")
    dml_effect = double_ml_estimate(W, T, Y)
    dml_error = abs(true_effect - dml_effect)
    print(f"    -> Double ML Estimated Effect: {dml_effect:.4f}")
    print(f"    -> Double ML Error:            {dml_error:.4f}  <-- FWL THEOREM RECOVERS THE TRUTH")
    
    print("\n==========================================================")
    if dml_error < lasso_error:
        print("✅ CONCLUSION: Double ML successfully bypasses Regularization Bias.")
    else:
        print("❌ WARNING: Unexpected result.")
