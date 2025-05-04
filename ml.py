# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math  # For potential calculations if needed elsewhere
import warnings  # To suppress potential TabPFN warnings if needed
from matplotlib import rcParams
import matplotlib as mpl
import os # For creating output directory if needed
import re # For cleaning filenames

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Import standard regressors
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
# from sklearn.svm import SVR

# Import TabPFN Regressor
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
except ImportError:
    print("Warning: tabpfn not found. Install it with 'pip install tabpfn'. TabPFN model will be skipped.")
    tabpfn_available = False

# Configure matplotlib for publication-quality figures (Nature style)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = True # Set to False if you want white background saved
plt.rcParams['legend.frameon'] = False

# Configure matplotlib percentage formatting
from matplotlib.ticker import PercentFormatter

# --- 1. Load Data ---
csv_path = 'suzuki_rdkit_features.csv'  # <<< IMPORTANT: Replace with your CSV file path
try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded data from: {csv_path}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 2. Data Preprocessing ---
if df.shape[1] < 2:
    print("Error: Data must have at least two columns (one feature, one target).")
    exit()

target_column_name = 'target_yield'  # <<< CHANGE THIS if your target column is different
if target_column_name not in df.columns:
    print(f"Warning: Target column '{target_column_name}' not found. Checking last column.")
    original_last_col = df.columns[-1]
    if original_last_col != target_column_name:
        print(f"Renaming last column '{original_last_col}' to '{target_column_name}' for consistency.")
        df = df.rename(columns={original_last_col: target_column_name})
    else:
        print(f"Last column is already named '{target_column_name}'.")

if target_column_name not in df.columns:
    print(f"Error: Target column '{target_column_name}' still not found after checks.")
    exit()

X = df.drop(columns=[target_column_name])
y = df[target_column_name]

# --- Feature Selection / Handling ---
num_cols_before = X.shape[1]
X = X.select_dtypes(include=np.number)
if X.shape[1] < num_cols_before:
    print(f"Warning: Dropped {num_cols_before - X.shape[1]} non-numeric feature columns.")

MAX_FEATURES_FOR_TABPFN = 100
if X.shape[1] > MAX_FEATURES_FOR_TABPFN and tabpfn_available:
    print(f"Warning: Dataset has {X.shape[1]} features, which exceeds the typical limit ({MAX_FEATURES_FOR_TABPFN}) for TabPFN.")
    print(f"Using only the first {MAX_FEATURES_FOR_TABPFN} features for all models to accommodate TabPFN.")
    X = X.iloc[:, :MAX_FEATURES_FOR_TABPFN]

if X.isnull().values.any():
    print("Warning: NaN values found in features (X). Imputing with column means.")
    X = X.fillna(X.mean())

if X.empty:
    print("Error: No numeric feature columns remaining after preprocessing.")
    exit()

if not pd.api.types.is_numeric_dtype(y):
    print(f"Warning: Target column '{target_column_name}' is not numeric. Attempting conversion.")
    y = pd.to_numeric(y, errors='coerce')
if y.isnull().values.any():
    print("Warning: NaN values found in target (y) after conversion/initially. Imputing with mean.")
    mean_y = y.mean()
    if pd.isna(mean_y):
        print("Error: Cannot compute mean of target variable (possibly all NaNs?). Exiting.")
        exit()
    y = y.fillna(mean_y)

print(f"\nData shape - Features (X): {X.shape}, Target (y): {y.shape}")

# --- Check Dataset Size for TabPFN ---
MAX_SAMPLES_FOR_TABPFN = 10000
if X.shape[0] > MAX_SAMPLES_FOR_TABPFN * 2 and tabpfn_available:
    print(f"Warning: Dataset has {X.shape[0]} samples, which might be large for standard TabPFN (designed for ~{MAX_SAMPLES_FOR_TABPFN} training samples).")

# --- 3. Split Data ---
test_set_size = 0.2
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_set_size, random_state=random_seed
)
print(f"Split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")

# --- Scaling Data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# --- 4. Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=random_seed),
    "KNeighbors": KNeighborsRegressor(n_jobs=-1),
    "Decision Tree": DecisionTreeRegressor(random_state=random_seed),
    "Random Forest": RandomForestRegressor(random_state=random_seed, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=random_seed),
    "AdaBoost": AdaBoostRegressor(random_state=random_seed),
    "Extra Trees": ExtraTreesRegressor(random_state=random_seed, n_jobs=-1),
}

if tabpfn_available:
    models["TabPFN"] = TabPFNRegressor(device='auto')
    print("Added TabPFN Regressor to the list of models.")

# --- 5. Train, Evaluate ---
print("\n--- Training and Evaluating Models ---")
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nProcessing model: {name}...")
    try:
        if name in ["Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost", "Extra Trees"]:
            X_train_fit = X_train
            X_test_predict = X_test
        elif name == "TabPFN":
            # Check if training data exceeds TabPFN limits AFTER split
            if X_train_scaled.shape[0] > MAX_SAMPLES_FOR_TABPFN:
                 print(f"  Warning for TabPFN: Training set size ({X_train_scaled.shape[0]}) exceeds recommended limit ({MAX_SAMPLES_FOR_TABPFN}).")
                 # Decide if you want to subsample or just proceed with warning
                 # Example subsampling (uncomment if desired):
                 # sample_indices = np.random.choice(X_train_scaled.index, size=MAX_SAMPLES_FOR_TABPFN, replace=False)
                 # X_train_fit = X_train_scaled.loc[sample_indices]
                 # y_train_fit = y_train.loc[sample_indices]
                 # print(f"  Subsampled training data for TabPFN to {MAX_SAMPLES_FOR_TABPFN} samples.")
            #else: # No subsampling needed or chosen
            X_train_fit = X_train_scaled
            X_test_predict = X_test_scaled
            y_train_fit = y_train # Use original y_train unless subsampled
        else:  # Linear, KNN, Ridge etc.
            X_train_fit = X_train_scaled
            X_test_predict = X_test_scaled
            y_train_fit = y_train # Use original y_train

        # Fit model
        # Need to handle potential subsampling for y_train with TabPFN
        if name == "TabPFN" and 'y_train_fit' in locals():
             model.fit(X_train_fit, y_train_fit)
        else:
             model.fit(X_train_fit, y_train)


        # Predict
        y_pred = model.predict(X_test_predict)

        if isinstance(y_pred, tuple): # Handle cases like TabPFN which might return tuple
            y_pred = y_pred[0]
        y_pred = np.squeeze(y_pred)  # Ensure it's 1D

        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
        results[name] = model_metrics
        predictions[name] = y_pred

        print(f"  {name} - Test Set Performance:")
        print(f"    MSE : {mse:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE : {mae:.4f}")
        print(f"    R²  : {r2:.4f}")

    except Exception as e:
        print(f"  Error processing model {name}: {e}")
        if name == "TabPFN":
            print("      (TabPFN error might relate to dataset size (samples/features), memory, or device setting ('cpu'/'cuda'))")
        results[name] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan}
        predictions[name] = None

# --- 6. Generate Individual Scatter Plots ---
print("\n--- Generating Publication-Quality Scatter Plots (Individual Files) ---")

valid_models = {name: pred for name, pred in predictions.items() if pred is not None}
num_valid_models = len(valid_models)
output_directory = "model_performance_plots" # Define a folder to save plots

if not os.path.exists(output_directory):
    try:
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    except Exception as e:
        print(f"Warning: Could not create output directory '{output_directory}': {e}. Saving plots in current directory.")
        output_directory = "." # Save in current directory if creation fails

if num_valid_models > 0:
    # Determine shared plot limits for consistency across plots
    all_valid_preds_flat = [p for p in valid_models.values() if p is not None]
    if not all_valid_preds_flat:
        print("Error: No valid predictions available to determine plot limits.")
        all_preds_np = np.array([np.nan])
    else:
        # Ensure concatenation works even with single model
        if len(all_valid_preds_flat) > 1:
             all_preds_np = np.concatenate(all_valid_preds_flat)
        elif len(all_valid_preds_flat) == 1:
             all_preds_np = np.array(all_valid_preds_flat[0])
        else: # Should not happen due to check above, but safety
             all_preds_np = np.array([np.nan])


    valid_y_test = y_test.dropna()
    valid_preds = all_preds_np[~np.isnan(all_preds_np)]

    if len(valid_y_test) == 0 or len(valid_preds) == 0:
        min_val, max_val = 0, 1 # Default limits
        print("Warning: Could not determine plot limits from data (NaNs?). Using default [0, 1].")
    else:
        min_val = min(valid_y_test.min(), valid_preds.min())
        max_val = max(valid_y_test.max(), valid_preds.max())

    padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
    axis_min = min_val - padding
    axis_max = max_val + padding

    # Use a professional color palette (optional, can use default or single color)
    cmap = plt.cm.viridis # Or plt.cm.Blues, plt.cm.Greys etc.
    # If you want distinct colors per plot:
    # colors = cmap(np.linspace(0.1, 0.9, num_valid_models))
    # Or a single color for all plots:
    plot_color = cmap(0.6) # A nice color from the chosen map

    plot_index = 0 # Keep index if using distinct colors
    for name, y_pred in valid_models.items():
        if y_pred is not None:
            # --- Create a NEW figure and axes for EACH model ---
            fig_single, ax_single = plt.subplots(figsize=(5, 4.5)) # Adjust size as needed

            model_metrics = results[name]
            # Use distinct color if desired: color = colors[plot_index]
            # Use single color:
            color = plot_color

            y_pred_np = np.asarray(y_pred)
            mask = ~np.isnan(y_test) & ~np.isnan(y_pred_np)
            y_test_plot = y_test[mask]
            y_pred_plot = y_pred_np[mask]

            if len(y_test_plot) > 0:
                # Create beautiful scatter plot
                scatter = ax_single.scatter(
                    y_test_plot, y_pred_plot,
                    alpha=0.7,
                    s=40,
                    edgecolor='white',
                    linewidth=0.5,
                    c=[color], # Note: c expects a list even for single color
                    marker='o'
                )

            # Add the identity line (y=x)
            identity_line = ax_single.plot(
                [axis_min, axis_max], [axis_min, axis_max],
                '--',
                color='#E41A1C', # Professional red color
                linewidth=1.0,
                label='_nolegend_' # Hide from potential legend
            )

            # Add metrics text
            metrics_text = (
                f"$R^2$ = {model_metrics.get('R²', float('nan')):.3f}\n"
                f"RMSE = {model_metrics.get('RMSE', float('nan')):.1f}%\n"
                f"MAE = {model_metrics.get('MAE', float('nan')):.1f}%"
            )
            ax_single.annotate(
                metrics_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="lightgray",
                    alpha=0.9,
                    linewidth=0.8
                ),
                ha='left',
                va='top',
                fontsize=8,
                color='#333333'
            )

            # Set title and INDIVIDUAL axis labels
            ax_single.set_title(f'{name} Performance', fontweight='medium', pad=10)
            ax_single.set_xlabel('Actual Yield (%)', fontsize=10, fontweight='medium')
            ax_single.set_ylabel('Predicted Yield (%)', fontsize=10, fontweight='medium')

            # Set axis limits (using pre-calculated shared limits)
            ax_single.set_xlim(axis_min, axis_max)
            ax_single.set_ylim(axis_min, axis_max)

            # Format axis tick labels to show percentages
            if max_val <= 1.1 : # Allow slight overshoot if data near 1
                # Values are likely 0-1 range
                ax_single.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
                ax_single.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
                # Adjust RMSE/MAE labels if needed (they are already printed as %)
            else:
                # Values are likely already in percentage (0-100 range)
                ax_single.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
                ax_single.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
                # Adjust RMSE/MAE labels if needed (e.g. remove %)
                # metrics_text = ( f"$R^2$ = ...\nRMSE = ...\nMAE = ..." ) # Without %

            # Remove grid
            ax_single.grid(False)

            # Clean model name for filename (remove spaces, special chars)
            safe_name = re.sub(r'[^\w\-_.]', '_', name) # Keep word chars, hyphen, underscore, dot
            filename = os.path.join(output_directory, f'performance_{safe_name}.png')

            # Adjust layout for the single figure
            plt.tight_layout()

            # Save the INDIVIDUAL figure
            try:
                plt.savefig(
                    filename,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white', # Ensure non-transparent background if needed
                    edgecolor='none'
                )
                print(f"  Successfully saved plot: {filename}")
            except Exception as e:
                print(f"  Error saving plot {filename}: {e}")

            # --- Close the current figure to free memory ---
            # plt.show() # Uncomment if you want to see each plot interactively
            plt.close(fig_single)

            plot_index += 1 # Increment if using distinct colors

else:
    print("\nNo models produced valid predictions to plot.")

# --- 7. Optional: Display Summary Table ---
if results:
    print("\n--- Overall Performance Summary ---")
    valid_results = {k: v for k, v in results.items() if not np.isnan(v.get('R²', np.nan))}
    if valid_results:
        results_df = pd.DataFrame.from_dict(valid_results, orient='index')
        results_df.index.name = 'Model'
        results_df = results_df.sort_values(by='R²', ascending=False)
        print(results_df.round(4))
    else:
        print("No models were successfully evaluated.")
else:
    print("\nNo models were processed.")

print("\n--- Script Execution Complete ---")
