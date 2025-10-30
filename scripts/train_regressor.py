#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train regression models to predict algorithm iterations from log features.

Usage:
    python train_regressor.py <input.pkl> --regressor <type> [options]
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


AVAILABLE_REGRESSORS = [
    'linear',
    'poly2', 'poly3', 'poly4',
    'xgboost',
    'lightgbm',
    'random_forest',
    'gradient_boosting'
]

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# Edit this list to exclude specific features from training
# Leave empty to use all features (except 'file' and 'iter')
# ============================================================================
FEATURES_TO_EXCLUDE = [
    # Example usage (uncomment to exclude):
    # 'time',
    # 'avg_constraint_range',
    # 'binary_ratio',
    'avg_obj_coeff_magnitude',
    'n_of_minimums_for_exit',
    'feasibility_run',
    'fixed_var_ratio',
    'unbounded_var_ratio',
    'obj_var_ratio',
    'avg_related_vars_per_var',
    'avg_constraint_range',
    'nnz_variance',
    'avg_variable_range',
    'min_nnz_per_row',
    'constraint_var_ratio',
    'avg_var_degree',
    'equality_ratio',
    'integer_ratio',
    'binary_ratio',
    'max_related_vars',
    'problem_size_score',
    'structural_complexity',
    'tight_constraint_ratio'
]

# Alternatively, specify ONLY the features you want to use
# If non-empty, only these features will be used (overrides FEATURES_TO_EXCLUDE)
FEATURES_TO_INCLUDE_ONLY = [
    # Example usage (uncomment to use only specific features):
    # 'n_variables',
    # 'n_constraints',
    # 'sparsity',
]
# ============================================================================


def load_data(pkl_path: str) -> pd.DataFrame:
    """Load pickle file and convert to DataFrame."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of dictionaries, got {type(data)}")

    if len(data) == 0:
        raise ValueError("Empty dataset")

    df = pd.DataFrame(data)

    # Validate required columns
    if 'file' not in df.columns:
        raise ValueError("Missing required 'file' column in data")
    if 'iter' not in df.columns:
        raise ValueError("Missing required 'iter' column in data")

    return df


def split_by_files(df: pd.DataFrame, test_size: float = 0.2,
                   random_state: int = None, stratify_by: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets based on unique files.
    Ensures all entries from a file go to either train or test, not both.

    Args:
    ----
        stratify_by: Optional column name to stratify split (e.g., 'iter' for balanced target distribution)
    """
    unique_files = df['file'].unique()

    # Optionally stratify by target distribution
    if stratify_by:
        # Create stratification labels based on quantiles of the specified column
        file_stats = df.groupby('file')[stratify_by].median()
        stratify_labels = pd.qcut(file_stats, q=min(5, len(unique_files)), labels=False, duplicates='drop')
        train_files, test_files = train_test_split(
            unique_files,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
    else:
        train_files, test_files = train_test_split(
            unique_files,
            test_size=test_size,
            random_state=random_state
        )

    train_df = df[df['file'].isin(train_files)].copy()
    test_df = df[df['file'].isin(test_files)].copy()

    print(f"\nData Split:")
    print(f"  Total entries: {len(df)}")
    print(f"  Train entries: {len(train_df)} ({len(train_files)} files)")
    print(f"  Test entries: {len(test_df)} ({len(test_files)} files)")

    # Check distribution similarity
    train_target_mean = train_df['iter'].mean()
    test_target_mean = test_df['iter'].mean()
    train_target_std = train_df['iter'].std()
    test_target_std = test_df['iter'].std()

    print(f"\nTarget ('iter') Distribution:")
    print(f"  Train: mean={train_target_mean:.2f}, std={train_target_std:.2f}")
    print(f"  Test:  mean={test_target_mean:.2f}, std={test_target_std:.2f}")

    mean_diff_pct = abs(train_target_mean - test_target_mean) / train_target_mean * 100
    if mean_diff_pct > 10:
        print(f"  ⚠️  Warning: Train/test target means differ by {mean_diff_pct:.1f}%")
        print(f"      Consider using stratified split or different random seed")

    return train_df, test_df


def list_available_features(df: pd.DataFrame) -> List[str]:
    """
    List all available numeric features in the dataset.
    Helper function to see what features can be selected/excluded.
    """
    X = df.drop(columns=['iter', 'file'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    return sorted(X.columns.tolist())


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target from DataFrame.
    Excludes 'file' and 'iter' from features.
    Applies feature selection based on FEATURES_TO_EXCLUDE and FEATURES_TO_INCLUDE_ONLY.
    """
    # Separate target
    y = df['iter'].copy()

    # Drop non-feature columns
    X = df.drop(columns=['iter', 'file'])

    # Ensure all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"Warning: Dropping non-numeric columns: {non_numeric}")
        X = X.select_dtypes(include=[np.number])

    # Apply feature selection
    original_feature_count = len(X.columns)

    if FEATURES_TO_INCLUDE_ONLY:
        # Use only specified features
        available_features = [f for f in FEATURES_TO_INCLUDE_ONLY if f in X.columns]
        missing_features = [f for f in FEATURES_TO_INCLUDE_ONLY if f not in X.columns]

        if missing_features:
            print(f"Warning: Requested features not found in data: {missing_features}")

        X = X[available_features]
        print(f"Feature selection: Using only {len(available_features)} specified features")

    elif FEATURES_TO_EXCLUDE:
        # Exclude specified features
        features_to_drop = [f for f in FEATURES_TO_EXCLUDE if f in X.columns]
        if features_to_drop:
            X = X.drop(columns=features_to_drop)
            print(f"Feature selection: Excluded {len(features_to_drop)} features: {features_to_drop}")

    feature_names = X.columns.tolist()

    if len(feature_names) == 0:
        raise ValueError(
            "No features remaining after feature selection! "
            "Check FEATURES_TO_EXCLUDE and FEATURES_TO_INCLUDE_ONLY settings."
        )

    if len(feature_names) != original_feature_count:
        print(f"  Using {len(feature_names)} of {original_feature_count} available features")

    return X, y, feature_names


def create_regressor(regressor_type: str, random_state: int = None,
                     tune_hyperparams: bool = False, verbose: bool = True):
    """
    Create a regression model with optional preprocessing pipeline.

    Returns: (model, needs_scaling)
    """
    if regressor_type == 'linear':
        model = LinearRegression()
        needs_scaling = True

    elif regressor_type.startswith('poly'):
        degree = int(regressor_type[-1])
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', Ridge(alpha=1.0))  # Ridge to handle multicollinearity
        ])
        needs_scaling = True

    elif regressor_type == 'xgboost':
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        params = {
            'objective': 'reg:squarederror',
            'random_state': random_state,
            'n_estimators': 100,
            'max_depth': 6,
            'tree_method': 'hist',
            'learning_rate': 0.1,
            'verbosity': 1 if verbose else 0,
            # Regularization to prevent overfitting
            'min_child_weight': 3,      # Minimum sum of weights in a leaf
            'gamma': 0.1,                # Minimum loss reduction for split
            'subsample': 0.8,            # Fraction of samples per tree
            'colsample_bytree': 0.8,     # Fraction of features per tree
            'reg_alpha': 0.1,            # L1 regularization
            'reg_lambda': 1.0,           # L2 regularization
        }

        if tune_hyperparams:
            # Stronger regularization for tuned version
            params.update({
                'n_estimators': 200,
                'max_depth': 5,              # Shallower trees
                'learning_rate': 0.05,       # Lower learning rate
                'min_child_weight': 5,       # Higher minimum weight
                'gamma': 0.2,                # More conservative splits
                'subsample': 0.7,            # More aggressive subsampling
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,            # Stronger L1
                'reg_lambda': 2.0,           # Stronger L2
            })

        model = xgb.XGBRegressor(**params)
        needs_scaling = False

    elif regressor_type == 'lightgbm':
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        params = {
            'objective': 'regression',
            'random_state': random_state,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'verbosity': 1 if verbose else -1,
            # Regularization to prevent overfitting
            'min_child_weight': 3,
            'min_split_gain': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }

        if tune_hyperparams:
            # Stronger regularization for tuned version
            params.update({
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'min_child_weight': 5,
                'min_split_gain': 0.2,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 2.0,
            })

        model = lgb.LGBMRegressor(**params)
        needs_scaling = False

    elif regressor_type == 'random_forest':
        params = {
            'random_state': random_state,
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'verbose': 1 if verbose else 0
        }

        if tune_hyperparams:
            params.update({
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5
            })

        model = RandomForestRegressor(**params)
        needs_scaling = False

    elif regressor_type == 'gradient_boosting':
        params = {
            'random_state': random_state,
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'verbose': 1 if verbose else 0
        }

        if tune_hyperparams:
            params.update({
                'n_estimators': 200,
                'max_depth': 7,
                'learning_rate': 0.05
            })

        model = GradientBoostingRegressor(**params)
        needs_scaling = False

    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")

    return model, needs_scaling


def get_feature_importance(model, feature_names: List[str],
                          regressor_type: str) -> None:
    """Extract and print feature importance if available."""
    print(f"\nFeature Importance:")

    try:
        if regressor_type in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']:
            # Tree-based models have feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            for i, idx in enumerate(indices, 1):
                print(f"  {i:3d}. {feature_names[idx]:40s}: {importances[idx]:.6f}")

        elif regressor_type == 'linear':
            # Linear regression coefficients
            coefs = np.abs(model.coef_)
            indices = np.argsort(coefs)[::-1]

            for i, idx in enumerate(indices, 1):
                print(f"  {i:3d}. {feature_names[idx]:40s}: {coefs[idx]:.6f}")

        elif regressor_type.startswith('poly'):
            # For polynomial, get feature names and coefficients from the Ridge step
            poly_features = model.named_steps['poly'].get_feature_names_out(feature_names)
            coefs = np.abs(model.named_steps['regressor'].coef_)
            indices = np.argsort(coefs)[::-1]

            print(f"  (Showing top 50 of {len(indices)} polynomial features)")
            for i, idx in enumerate(indices[:50], 1):
                feat_name = poly_features[idx]
                # Truncate very long polynomial feature names
                if len(feat_name) > 60:
                    feat_name = feat_name[:57] + "..."
                print(f"  {i:3d}. {feat_name:60s}: {coefs[idx]:.6f}")
        else:
            print("  Feature importance not available for this model type")

    except Exception as e:
        print(f"  Could not extract feature importance: {e}")


def evaluate_model(model, X_train, y_train, X_test, y_test,
                   feature_names: List[str], regressor_type: str,
                   cv_folds: int = 5, verbose: int = 0,
                   skip_cv: bool = False, X_test_original: pd.DataFrame = None,
                   test_df: pd.DataFrame = None) -> Tuple[float, float]:
    """Evaluate model and print metrics. Returns (train_r2, test_r2).

    Args:
    ----
        X_test_original: Unscaled X_test for displaying feature values
        test_df: Original test dataframe with 'file' column
    """
    # Cross-validation on training set (skip if using early stopping)
    if not skip_cv:
        print(f"\nCross-Validation on Training Set ({cv_folds}-fold):")
        try:
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv_folds,
                                        scoring='neg_mean_squared_error',
                                        n_jobs=-1,
                                        verbose=verbose)
            cv_rmse = np.sqrt(-cv_scores)
            print(f"  CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
        except Exception as e:
            print(f"  CV failed (likely due to early stopping): {str(e)[:100]}")
            print(f"  Skipping cross-validation...")
    else:
        print(f"\nSkipping cross-validation (incompatible with early stopping)")

    # Training set metrics
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    print(f"\nTraining Set Metrics:")
    print(f"  MSE:  {train_mse:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")

    # Test set metrics
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {test_mse:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    # Feature importance
    get_feature_importance(model, feature_names, regressor_type)

    # Sample predictions
    print(f"\n20 Sample Predictions from Test Set:")
    print(f"  {'Actual':>10s}  {'Predicted':>10s}  {'Error':>10s}  {'Error %':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    sample_indices = np.random.choice(len(y_test), min(20, len(y_test)), replace=False)
    for idx in sample_indices:
        actual = y_test.iloc[idx]
        predicted = y_test_pred[idx]
        error = actual - predicted
        error_pct = (error / actual * 100) if actual != 0 else 0
        print(f"  {actual:10.2f}  {predicted:10.2f}  {error:10.2f}  {error_pct:9.2f}%")

    # Show worst predictions with feature values
    print(f"\n5 Worst Predictions (Largest Absolute Error):")
    abs_errors = np.abs(y_test_pred - y_test.values)
    worst_indices = np.argsort(abs_errors)[-5:][::-1]

    # Use original (unscaled) features if available, otherwise use X_test
    X_display = X_test_original if X_test_original is not None else X_test

    for rank, idx in enumerate(worst_indices, 1):
        actual = y_test.iloc[idx]
        predicted = y_test_pred[idx]
        error = actual - predicted
        error_pct = (error / actual * 100) if actual != 0 else 0

        # Get filename if available
        filename = ""
        if test_df is not None and 'file' in test_df.columns:
            filename = f" (file: {test_df.iloc[idx]['file']})"

        print(f"\n  #{rank} - Actual: {actual:.2f}, Predicted: {predicted:.2f}, "
              f"Error: {error:.2f} ({error_pct:.1f}%){filename}")

        # Get feature values (handle both DataFrame and array)
        if isinstance(X_display, pd.DataFrame):
            feature_values = X_display.iloc[idx].values
        elif isinstance(X_display, np.ndarray):
            feature_values = X_display[idx]
        else:
            feature_values = X_display[idx]

        # Display features compactly (5 per line)
        print(f"      Features:", end="")
        for i, (feat_name, feat_val) in enumerate(zip(feature_names, feature_values)):
            if i % 5 == 0:
                print(f"\n        ", end="")
            # Format feature value
            if isinstance(feat_val, (int, np.integer)):
                print(f"{feat_name}={feat_val}", end="  ")
            else:
                print(f"{feat_name}={feat_val:.3g}", end="  ")
        print()  # Final newline

    return train_r2, test_r2


def compile_model_treelite(model, regressor_type: str, output_dir: str,
                          num_threads: int, X_train=None,
                          annotate: bool = False, quantize: bool = False,
                          feature_names: List[str] = None,
                          model_name: str = None) -> None:
    """Compile XGBoost/LightGBM model to C source files using TL2cgen.

    Args:
    ----
        model: Trained model
        regressor_type: Type of regressor
        output_dir: Output directory
        num_threads: Number of parallel compilation threads
        X_train: Training data for branch annotation (optional)
        annotate: Whether to annotate branches for optimization
        quantize: Whether to use quantization in code generation
        feature_names: List of feature names in expected order (optional)
        model_name: Name prefix for functions (optional, derived from training file)
    """
    if regressor_type not in ['xgboost', 'lightgbm']:
        print(f"Warning: TL2cgen compilation only supported for XGBoost and LightGBM, skipping for {regressor_type}")
        return

    try:
        import treelite
        import tl2cgen
    except ImportError as e:
        missing = []
        try:
            import treelite
        except ImportError:
            missing.append("treelite")
        try:
            import tl2cgen
        except ImportError:
            missing.append("tl2cgen")

        print(f"Warning: {', '.join(missing)} not installed. Install with: pip install {' '.join(missing)}")
        print("Skipping C code generation.")
        return

    optimization_info = []
    if annotate:
        optimization_info.append("branch annotation")
    if quantize:
        optimization_info.append("quantization")

    opt_str = f" with {', '.join(optimization_info)}" if optimization_info else ""
    print(f"\nGenerating C source code with TL2cgen (threads={num_threads}){opt_str}...")

    # Convert model to treelite format using frontend API
    try:
        if regressor_type == 'xgboost':
            tl_model = treelite.frontend.from_xgboost(model.get_booster())
        elif regressor_type == 'lightgbm':
            tl_model = treelite.frontend.from_lightgbm(model.booster_)
    except Exception as e:
        print(f"Warning: Failed to convert {regressor_type} model to treelite: {e}")
        return

    # Annotate branches if requested and training data is available
    annotation_path = None
    if annotate and X_train is not None:
        try:
            print("  Annotating branches with training data...")
            # Convert to numpy array if it's a DataFrame
            if hasattr(X_train, 'values'):
                X_train_array = X_train.values.astype(np.float32)
            else:
                X_train_array = np.asarray(X_train, dtype=np.float32)

            dmat = tl2cgen.DMatrix(X_train_array, dtype='float32')
            annotation_path = os.path.join(output_dir, f'{regressor_type}_annotation.json')
            tl2cgen.annotate_branch(tl_model, dmat=dmat, path=annotation_path, verbose=False)
            print(f"  Branch annotations saved to: {annotation_path}")
        except Exception as e:
            print(f"  Warning: Branch annotation failed: {e}")
            print("  Continuing without branch annotation")
            annotation_path = None
    elif annotate and X_train is None:
        print("  Warning: Branch annotation requested but no training data available")
        print("  Skipping branch annotation")

    # Generate C source files using TL2cgen
    source_dir = os.path.join(output_dir, f'{regressor_type}_c_code')

    try:
        #params = {'parallel_comp': num_threads}
        params = {}

        # Add quantization parameter if requested
        if quantize:
            params['quantize'] = 1  # Enable quantization in code generation

        # Add annotation file if available
        if annotation_path:
            params['annotate_in'] = annotation_path

        tl2cgen.generate_c_code(
            tl_model,
            dirpath=source_dir,
            params=params,
            verbose=False
        )

        # Post-process generated files
        header_path = os.path.join(source_dir, 'header.h')
        main_path = os.path.join(source_dir, 'main.c')
        quantize_path = os.path.join(source_dir, 'quantize.c')
        recipe_path = os.path.join(source_dir, 'recipe.json')

        # Rename all .c files to .cpp and wrap in class
        if model_name:
            try:
                import glob
                c_files = glob.glob(os.path.join(source_dir, '*.c'))

                for c_file in c_files:
                    cpp_file = c_file[:-2] + '.cpp'

                    # Read content
                    with open(c_file, 'r') as f:
                        content = f.read()

                    # Split content into includes and rest
                    lines = content.split('\n')
                    include_lines = []
                    code_lines = []
                    in_includes = True

                    for line in lines:
                        if in_includes and (line.strip().startswith('#include') or line.strip().startswith('#') or line.strip() == ''):
                            include_lines.append(line)
                        else:
                            in_includes = False
                            code_lines.append(line)

                    # Prefix function definitions with ClassName:: (for .cpp files, not class wrapping)
                    import re
                    processed_lines = []
                    for line in code_lines:
                        # Detect function definitions (return_type function_name(...))
                        if line and not line.strip().startswith('//') and not line.strip().startswith('/*'):
                            # Check if it's a function definition
                            # Pattern: type name(...) or type* name(...) etc.
                            func_pattern = r'^(\s*)((?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?[\w_]+(?:\s*\*)*\s+)([\w_]+)(\s*\()'
                            match = re.match(func_pattern, line)
                            if match and '::' not in line:  # Don't add if already qualified
                                indent = match.group(1)
                                return_type = match.group(2)
                                func_name = match.group(3)
                                rest = line[match.end(3):]
                                # Prefix function name with class name
                                line = f'{indent}{return_type}{model_name}::{func_name}{rest}'
                        processed_lines.append(line)
                    code_lines = processed_lines

                    # Don't wrap in class for .cpp files - just output the definitions
                    includes_str = '\n'.join(include_lines)
                    code_str = '\n'.join(code_lines)

                    # For .cpp files, no class wrapper needed
                    cpp_content = f'{includes_str}\n\n{code_str}\n'

                    # Write to .cpp file
                    with open(cpp_file, 'w') as f:
                        f.write(cpp_content)

                    # Remove original .c file
                    os.remove(c_file)

                # Update paths for further processing
                main_path = main_path[:-2] + '.cpp'
                quantize_path = quantize_path[:-2] + '.cpp'

                print(f"  Renamed {len(c_files)} .c files to .cpp")
            except Exception as e:
                print(f"  Warning: Failed to rename .c files: {e}")

        # Optimize main.cpp by removing unnecessary missing data checks
        # Since all features are always provided, replace !(data[X].missing != -1) with false
        if os.path.exists(main_path):
            try:
                with open(main_path, 'r') as f:
                    content = f.read()

                # Replace pattern !(data[N].missing != -1) with false
                import re
                original_content = content
                content = re.sub(r'!\(data\[\d+\]\.missing != -1\)', 'false', content)

                if content != original_content:
                    with open(main_path, 'w') as f:
                        f.write(content)
                    print(f"  Optimized main.cpp by removing unnecessary missing data checks")
            except Exception as e:
                print(f"  Warning: Failed to optimize main.cpp: {e}")

        # Wrap header.h content in class with #pragma once
        defines_to_move = []
        if model_name and os.path.exists(header_path):
            try:
                with open(header_path, 'r') as f:
                    content = f.read()

                # Split content into includes, defines to move, and rest
                lines = content.split('\n')
                include_lines = []
                code_lines = []
                in_includes = True
                i = 0

                while i < len(lines):
                    line = lines[i]

                    if in_includes and (line.strip().startswith('#include') or line.strip() == ''):
                        include_lines.append(line)
                        i += 1
                    # Detect macros to move to main.cpp
                    elif line.strip().startswith('#if defined(__clang__)') or line.strip().startswith('#define N_TARGET') or line.strip().startswith('#define MAX_N_CLASS'):
                        in_includes = False
                        # Capture the entire #if block or single #define
                        if line.strip().startswith('#if defined(__clang__)'):
                            # Capture the entire #if...#endif block
                            macro_block = []
                            macro_block.append(line)
                            i += 1
                            while i < len(lines) and not lines[i].strip().startswith('#endif'):
                                macro_block.append(lines[i])
                                i += 1
                            if i < len(lines):
                                macro_block.append(lines[i])  # Include #endif
                                i += 1
                            defines_to_move.append('\n'.join(macro_block))
                        else:
                            # Single #define line
                            defines_to_move.append(line)
                            i += 1
                    else:
                        in_includes = False
                        code_lines.append(line)
                        i += 1

                # Add static keyword to function declarations
                import re
                processed_lines = []
                for line in code_lines:
                    # Detect function declarations/definitions (return_type function_name(...))
                    # Match lines that look like function declarations but don't already have static
                    if line and not line.strip().startswith('//') and not line.strip().startswith('/*'):
                        # Check if it's a function declaration/definition
                        # Pattern: type name(...) or type* name(...) or type name[...](...) etc.
                        func_pattern = r'^(\s*)((?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?[\w_]+(?:\s*\*)*\s+)([\w_]+)\s*\('
                        match = re.match(func_pattern, line)
                        if match and 'static' not in line:
                            indent = match.group(1)
                            return_type = match.group(2)
                            # Add static keyword
                            line = f'{indent}static {return_type}{line[len(indent)+len(return_type):]}'
                    processed_lines.append(line)
                code_lines = processed_lines

                # Wrap code in class declaration
                includes_str = '\n'.join(include_lines)
                code_str = '\n'.join(code_lines)

                wrapped_content = f'#pragma once\n\n{includes_str}\n\nclass {model_name} {{\npublic:\n{code_str}\n}};  // class {model_name}\n'

                with open(header_path, 'w') as f:
                    f.write(wrapped_content)

                print(f"  Wrapped header.h in class '{model_name}' with #pragma once")
            except Exception as e:
                print(f"  Warning: Failed to wrap header.h: {e}")

        # Add defines to main.cpp (moved from header.h)
        if defines_to_move and os.path.exists(main_path):
            try:
                with open(main_path, 'r') as f:
                    content = f.read()

                # Insert defines after includes (look for where code starts - typically after blank line after includes)
                defines_str = '\n'.join(defines_to_move)

                # Find the first non-include, non-blank line to insert before
                lines = content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith('#include'):
                        insert_pos = i
                        break

                # Insert defines at the position
                lines.insert(insert_pos, defines_str)
                lines.insert(insert_pos + 1, '')  # Add blank line after defines

                content = '\n'.join(lines)
                with open(main_path, 'w') as f:
                    f.write(content)
                print(f"  Moved {len(defines_to_move)} macro definition(s) from header.h to main.cpp")
            except Exception as e:
                print(f"  Warning: Failed to add defines to main.cpp: {e}")

        # Add feature names to header and implementation
        if feature_names and os.path.exists(header_path) and os.path.exists(main_path):
            try:
                # Append to header.h (inside class)
                with open(header_path, 'r') as f:
                    content = f.read()

                # Insert before closing class
                insertion = f'\n    // Feature names\n    static constexpr int NUM_FEATURES = {len(feature_names)};\n    static const char* feature_names[NUM_FEATURES];\n'
                content = content.replace(f'}};  // class {model_name}\n', f'{insertion}}};  // class {model_name}\n')

                with open(header_path, 'w') as f:
                    f.write(content)

                # Append to main.cpp (at the end of the file, outside any class)
                with open(main_path, 'r') as f:
                    content = f.read()

                # Append feature array definition at the end of the file
                feature_array = f'\n// Feature names array\nconst char* {model_name}::feature_names[{model_name}::NUM_FEATURES] = {{\n'
                for i, name in enumerate(feature_names):
                    comma = ',' if i < len(feature_names) - 1 else ''
                    feature_array += f'    "{name}"{comma}\n'
                feature_array += '};\n'

                # Append to end of file
                content = content.rstrip() + '\n' + feature_array

                with open(main_path, 'w') as f:
                    f.write(content)

                print(f"  Added {len(feature_names)} feature names to header.h and main.cpp")
            except Exception as e:
                print(f"  Warning: Failed to add feature names: {e}")

        # Remove recipe.json if it exists
        if os.path.exists(recipe_path):
            try:
                os.remove(recipe_path)
                print(f"  Removed recipe.json")
            except Exception as e:
                print(f"  Warning: Failed to remove recipe.json: {e}")

        opt_msg = []
        if annotation_path:
            opt_msg.append("branch-annotated")
        if quantize:
            opt_msg.append("quantized")
        opt_suffix = f" ({', '.join(opt_msg)})" if opt_msg else ""

        print(f"C source code generated to: {source_dir}/")
        print(f"  Contains optimized model source code{opt_suffix} ready for compilation")
    except Exception as e:
        print(f"Warning: TL2cgen code generation failed: {e}")
        print("  Model saved in standard format only.")


def save_model(model, scaler, regressor_type: str, output_dir: str,
               feature_names: List[str]) -> None:
    """Save trained model and preprocessing components to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata = {
        'regressor_type': regressor_type,
        'feature_names': feature_names,
        'has_scaler': scaler is not None
    }

    metadata_path = os.path.join(output_dir, f'{regressor_type}_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\nSaved metadata to: {metadata_path}")

    # Save scaler if exists
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f'{regressor_type}_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to: {scaler_path}")

    # Save model
    if regressor_type == 'xgboost':
        # Save as UBJ with gzip compression
        model_path = os.path.join(output_dir, f'{regressor_type}_model.ubj')
        model.save_model(model_path)

        # Gzip the file
        import gzip
        import shutil
        with open(model_path, 'rb') as f_in:
            with gzip.open(model_path + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(model_path)  # Remove uncompressed version
        print(f"Saved model to: {model_path}.gz")
    elif regressor_type == 'lightgbm':
        # Save LightGBM model as text file
        model_path = os.path.join(output_dir, f'{regressor_type}_model.txt')
        model.booster_.save_model(model_path)
        print(f"Saved model to: {model_path}")
    else:
        # Save sklearn models as joblib (more efficient than pickle for large arrays)
        model_path = os.path.join(output_dir, f'{regressor_type}_model.joblib')
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train regression models to predict algorithm iterations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available regressors:
  linear            - Linear Regression
  poly2, poly3, poly4 - Polynomial Regression (degree 2, 3, 4)
  xgboost           - XGBoost Regressor
  lightgbm          - LightGBM Regressor
  random_forest     - Random Forest Regressor
  gradient_boosting - Gradient Boosting Regressor

Example:
  python train_regressor.py data.pkl --regressor xgboost --seed 42
  python train_regressor.py data.pkl --regressor lightgbm --seed 42
        """
    )

    parser.add_argument('input_pkl', help='Input pickle file with log data')
    parser.add_argument(
        '--regressor', '-r',
        required=True,
        choices=AVAILABLE_REGRESSORS,
        help='Type of regressor to train'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./models',
        help='Output directory for saved models (default: ./models)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (uses predefined tuned parameters)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of files to use for testing (default: 0.2)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable training progress output'
    )
    parser.add_argument(
        '--list-features',
        action='store_true',
        help='List all available features in the dataset and exit'
    )
    parser.add_argument(
        '--stratify-split',
        action='store_true',
        help='Stratify train/test split by target distribution (ensures balanced iter values)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=20,
        metavar='N',
        help='Enable early stopping for tree models (default: 20 rounds, use 0 to disable)'
    )
    parser.add_argument(
        '--treelite-compile',
        type=int,
        default=1,
        metavar='THREADS',
        help='Export XGBoost/LightGBM model as optimized C source code with TL2cgen (includes branch annotation and quantization)'
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Load data
    print(f"\nLoading data from: {args.input_pkl}")
    df = load_data(args.input_pkl)
    print(f"Loaded {len(df)} entries with {len(df.columns)} columns")

    # Extract model name from input file (for prefixing generated C functions)
    model_name = os.path.splitext(os.path.basename(args.input_pkl))[0]

    # If listing features, do that and exit
    if args.list_features:
        features = list_available_features(df)
        print(f"\n{'='*70}")
        print(f"Available features in dataset ({len(features)} total):")
        print(f"{'='*70}")
        for i, feat in enumerate(features, 1):
            print(f"  {i:3d}. {feat}")
        print(f"\nTo exclude features, edit FEATURES_TO_EXCLUDE in the script:")
        print(f"  {__file__}")
        print(f"\nTo use only specific features, edit FEATURES_TO_INCLUDE_ONLY")
        return

    # Split data by files
    stratify_by = 'iter' if args.stratify_split else None
    train_df, test_df = split_by_files(df, test_size=args.test_size,
                                       random_state=args.seed,
                                       stratify_by=stratify_by)

    # Prepare features
    X_train, y_train, feature_names = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    print(f"\nFeatures: {len(feature_names)}")
    print(f"Target: iter (predicting number of iterations)")

    # Create model
    print(f"\nTraining {args.regressor} regressor...")
    model, needs_scaling = create_regressor(
        args.regressor,
        random_state=args.seed,
        tune_hyperparams=args.tune,
        verbose=not args.no_progress
    )

    # Apply scaling if needed
    scaler = None
    X_test_original = X_test.copy()  # Keep unscaled version for display
    if needs_scaling:
        print("  Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train model
    if args.regressor.startswith('poly'):
        degree = int(args.regressor[-1])
        n_features = X_train_scaled.shape[1]
        from math import comb
        n_poly_features = sum(comb(n_features + d - 1, d) for d in range(1, degree + 1))
        print(f"  Generating {n_poly_features} polynomial features (degree {degree})...")

    # Use early stopping for tree-based models if requested
    if args.early_stopping and args.early_stopping > 0 and args.regressor in ['xgboost', 'lightgbm', 'gradient_boosting']:
        if args.regressor == 'xgboost':
            print(f"  Using early stopping (patience={args.early_stopping} rounds)...")
            # Set early stopping parameter
            model.set_params(early_stopping_rounds=args.early_stopping)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            # Report best iteration
            best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
            print(f"  Best iteration: {best_iteration} (out of {model.n_estimators} max)")
        elif args.regressor == 'lightgbm':
            print(f"  Using early stopping (patience={args.early_stopping} rounds)...")
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                callbacks=[
                    __import__('lightgbm').early_stopping(stopping_rounds=args.early_stopping, verbose=False)
                ]
            )
            # Report best iteration
            best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
            print(f"  Best iteration: {best_iteration} (out of {model.n_estimators} max)")
        else:  # gradient_boosting
            # Gradient Boosting uses n_iter_no_change parameter
            print(f"  Note: Use --tune with gradient_boosting for early stopping")
            model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train_scaled, y_train)

    print("  Training complete!")

    # Evaluate model
    skip_cv = args.early_stopping is not None and args.early_stopping > 0
    train_r2, test_r2 = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test,
                                       feature_names, args.regressor, cv_folds=args.cv_folds,
                                       verbose=2 if not args.no_progress else 0,
                                       skip_cv=skip_cv, X_test_original=X_test_original,
                                       test_df=test_df)

    # Save model
    save_model(model, scaler, args.regressor, args.output_dir, feature_names)

    # Compile with TL2cgen if requested (with optimizations enabled by default)
    if args.treelite_compile is not None:
        # Use unscaled training data for branch annotation
        # Only tree-based models (XGBoost, LightGBM) don't need scaling
        X_train_for_annotation = X_train if not needs_scaling else None

        compile_model_treelite(
            model,
            args.regressor,
            args.output_dir,
            args.treelite_compile,
            X_train=X_train_for_annotation,
            annotate=True,   # Always enable branch annotation
            quantize=True,   # Always enable quantization
            feature_names=feature_names,
            model_name=model_name
        )

    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print(f"\nFinal R² Scores:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")


if __name__ == '__main__':
    main()
