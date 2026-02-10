import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

class LeakageAuditor:
    """
    A rigorous audit tool to detect common data leakage patterns in 
    Quantitative Finance pipelines.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the auditor with the dataset.
        
        Args:
            df (pd.DataFrame): The dataframe containing features and targets.
                               Time should be the index or a column.
        """
        self.df = df

    def check_lookahead(self, 
                       features: List[str], 
                       target: str, 
                       threshold: float = 0.99) -> List[str]:
        """
        Checks if any feature is suspiciously correlated with the target,
        which often indicates look-ahead bias (e.g. including 'Close' in input
        when target is 'Next Day Close').
        
        Args:
            features (List[str]): List of feature column names.
            target (str): The target column name (e.g., 't+1 return').
            threshold (float): Correlation threshold to flag a feature.
            
        Returns:
            List[str]: List of flagged feature names.
        """
        flagged_features = []
        
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")
            
        y = self.df[target]
        
        for feature in features:
            if feature not in self.df.columns:
                continue
                
            # Calculate correlation
            # We use absolute correlation because a perfect inverse correlation 
            # is also a sign of leakage (or a trivial relationship).
            x = self.df[feature]
            corr = abs(x.corr(y))
            
            if corr >= threshold:
                flagged_features.append((feature, corr))
                print(f"[AUDIT FAIL] Feature '{feature}' has correlation {corr:.4f} with target '{target}'. Possible Leakage.")
            
        if not flagged_features:
            print("[AUDIT PASS] No direct look-ahead leakage correlations detected.")
            
        return [f[0] for f in flagged_features]

    def check_split_integrity(self, train_idx: pd.Index, test_idx: pd.Index) -> bool:
        """
        Verifies that the Train/Test split respects time ordering.
        Quantitative models must never shuffle data; Test must strictly follow Train.
        
        Args:
            train_idx: Index of training data.
            test_idx: Index of testing data.
            
        Returns:
            bool: True if split is valid (no future leakage), False otherwise.
        """
        max_train = train_idx.max()
        min_test = test_idx.min()
        
        if max_train >= min_test:
            print(f"[AUDIT FAIL] Temporal Leakage Detected!")
            print(f"Latest Train Timestamp: {max_train}")
            print(f"Earliest Test Timestamp: {min_test}")
            print("Test data appears before or overlaps with Training data.")
            return False
            
        print("[AUDIT PASS] Train/Test split respects temporal ordering.")
        return True

    def check_pca_leakage(self, 
                         scaler_fit_on_test: bool = False) -> bool:
        """
        Placeholder for checking if normalization/PCA statistics include test data.
        In a real pipeline, this would wrap the sklearn pipeline to verify 'fit' 
        was only called on X_train.
        """
        if scaler_fit_on_test:
             print("[AUDIT FAIL] Scaler/PCA appears to be fit on Test data (Distribution Leakage).")
             return False
        return True
