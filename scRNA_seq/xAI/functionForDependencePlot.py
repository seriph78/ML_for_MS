# +
# Cell 1
import os
import joblib
import pandas as pd

print(os.listdir("../Store")) # Check what's in the parent's Store directory

os.chdir("../Store")
print(f"Current working directory: {os.getcwd()}")

# +
microarray_expl=  joblib.load("./MICROARRAY/xgbDefFull_explSorted.pkl")
microarray_explanations = pd.DataFrame(microarray_expl.items(), columns=['Feature', 'MeanAbsoluteShap'])
microarray = pd.read_csv("./MICROARRAY/MergedDatasetFullCombatDeclustered_symbol.csv")
microarray_shapObject = joblib.load("./MICROARRAY/xgbDefFull_shapValues.pkl")

map_microarray = pd.read_csv("./MICROARRAY/patientMap.csv")
# Merge the dataframes on 'sampleID'
map_microarray["stage"] = map_microarray["Label"]
map_microarray.drop(["Label"], axis=1)
microarray_merged = pd.merge(microarray, map_microarray, on="SampleID", how="left")
print(map_microarray)
# -

import scanpy as sc
cd4_csf_expl = joblib.load("CD4_CSF_SAMU/xgbCSFBCELLS_feature_importance.pkl")
cd4_csf_explanations = pd.DataFrame(microarray_expl.items(), columns=['Feature', 'MeanAbsoluteShap'])
cd4_csf = sc.read_h5ad("CD4_CSF_SAMU/datasetDeclustered.h5ad")
cd4_csf_shapObject = joblib.load("CD4_CSF_SAMU/xgbCSFBCELLS shapexplainer.pkl")

import scanpy as sc
bcells_csf_expl = joblib.load("CSF_BCELLS/xgbCSFBCELLS_feature_importance.pkl")
bcells_csf_explanations = pd.DataFrame(microarray_expl.items(), columns=['Feature', 'MeanAbsoluteShap'])
bcells_csf = sc.read_h5ad("CSF_BCELLS/datasetDeclustered.h5ad")
bcells_csf_shapObject = joblib.load("CSF_BCELLS/xgbCSFBCELLS2 shapexplainer.pkl")

import scanpy as sc
bcells_pbmc_expl = joblib.load("PBMC_BCELLS_2DATASET/xgbPBMCBCELLS_feature_importance.pkl")
bcells_cpbmc_explanations = pd.DataFrame(microarray_expl.items(), columns=['Feature', 'MeanAbsoluteShap'])
bcells_pbmc = sc.read_h5ad("PBMC_BCELLS_2DATASET/datasetDeclustered.h5ad")
bcells_pbmc_shapObject = joblib.load("PBMC_BCELLS_2DATASET/xgbPBMCBCELLS shapexplainer.pkl")

# +
from sklearn.preprocessing import MinMaxScaler

# Funzione per il dependence plot (MODIFIED)
def dependence_plot(
    df_expl,
    shap_object,
    data,
    label_col,
    i,
    axes=None,
    type_data="microarray",
    genes=None,
    plot_title_suffix="",
    stage_col=None,  # <-- NEW: Optional parameter for detailed stages
):
    # Ensure genes is a list for consistent indexing later
    if genes is None and type_data == "singlecell":
        raise ValueError(
            "For 'singlecell' type_data, 'genes' parameter must be provided."
        )
    if genes is not None and not isinstance(genes, (list, pd.Index)):
        genes = list(genes)

    # Handle different data types for 'data'
    if isinstance(data, pd.DataFrame):
        current = data.copy()
        feature_names = current.columns
    elif type_data == "singlecell":
        current = pd.DataFrame(data, columns=genes)
        feature_names = genes
    else:
        raise TypeError(
            "Unsupported 'data' type. Must be pandas.DataFrame or compatible with 'singlecell' type_data."
        )

    # --- NEW: Logic to select the correct labels ---
    if stage_col is not None:
        # If detailed stages are provided, use them directly.
        labels = stage_col
    else:
        # Otherwise, fall back to the original binary MS/Control logic.
        labels = ["MS" if value == 1 else "Control" for value in label_col]

    # --- NEW: Expanded palette for all possible conditions ---
    palette = {
        "Control": "lightblue",
        "MS": "lightcoral",
        "PP": "mediumseagreen",
        "RR": "mediumpurple",
        "SP": "gold",
        "CIS": "darkorange",
    }

    top_features = df_expl.sort_values("MeanAbsoluteShap", ascending=False)[
        "Feature"
    ].values
    f1, f2 = top_features[:2]

    # Get index of feature for shap_object and original data
    idx_f1 = feature_names.get_loc(f1)
    idx_f2 = feature_names.get_loc(f2)

    # --- Plot for the first top feature (f1) ---
    shap_vals_f1 = shap_object.values[:, idx_f1]
    feature_vals1 = current.iloc[:, idx_f1].values

    customDf1 = pd.DataFrame(
        {"SHAP": shap_vals_f1, "Feature": feature_vals1, "Label": labels}
    )

    sns.scatterplot(
        data=customDf1, x="Feature", y="SHAP", hue="Label", palette=palette, ax=axes[0 + i]
    )
    axes[0 + i].set_title(
        f"Dependence plot for {f1} {plot_title_suffix}", fontsize=10
    )
    axes[0 + i].set_xlabel(f"{f1}")
    axes[0 + i].set_ylabel("SHAP value")
    axes[0 + i].legend()

    # --- Plot for the second top feature (f2) ---
    shap_vals_f2 = shap_object.values[:, idx_f2]
    feature_vals2 = current.iloc[:, idx_f2].values

    customDf2 = pd.DataFrame(
        {"SHAP": shap_vals_f2, "Feature": feature_vals2, "Label": labels}
    )

    sns.scatterplot(
        data=customDf2, x="Feature", y="SHAP", hue="Label", palette=palette, ax=axes[1 + i]
    )
    axes[1 + i].set_title(
        f"Dependence plot for {f2} {plot_title_suffix}", fontsize=10
    )
    axes[1 + i].set_xlabel(f"{f2}")
    axes[1 + i].set_ylabel("SHAP value")
    axes[1 + i].legend()


# +
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# Modifica globale delle dimensioni dei font
plt.rc("axes", titlesize=18, labelsize=16)   # titoli assi e figure
plt.rc("legend", fontsize=14, title_fontsize=16)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
# Initialize your figure and axes for all plots
fig, axes = plt.subplots(4, 2, figsize=(20, 20))
axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing




# --- Prepare data for the Microarray plot ---
# Load microarray data and the patient map
microarray = pd.read_csv("./MICROARRAY/MergedDatasetFullCombatDeclustered_symbol.csv")
map_microarray = pd.read_csv("./MICROARRAY/patientMap.csv")

# Rename the 'Label' column from the map file to 'stage'
map_microarray.rename(columns={"Label": "stage"}, inplace=True)

# Merge to get stage information
microarray_merged = pd.merge(
    microarray,
    map_microarray[["SampleID", "stage"]],
    on="SampleID",
    how="left",
)

# Create the final 'condition' column with detailed stages
microarray_merged["condition"] = np.where(
    microarray_merged["Label"] == 0,
    "Control",
    microarray_merged["stage"].fillna("MS"),
)

# --- Create the figure and call the plotting functions ---
fig, axes = plt.subplots(4, 2, figsize=(12, 20)) # 4 rows, 2 columns

# Call for Microarray (with the new 'stage_col' parameter)
dependence_plot(
    df_expl=microarray_explanations,
    shap_object=microarray_shapObject,
    data=microarray.drop(columns=["SampleID", "PatientID", "Label"]),
    label_col=microarray["Label"].values,
    i=0,
    axes=axes.flatten(), # Flatten axes for easier indexing
    plot_title_suffix="(Microarray)",
    stage_col=microarray_merged["condition"].values, # <-- PASS THE NEW STAGES HERE
)

# Calls for single-cell data (UNCHANGED)
dependence_plot(
    df_expl=cd4_csf_expl,
    shap_object=cd4_csf_shapObject,
    data=cd4_csf.X,
    label_col=cd4_csf.obs["condition"].apply(lambda x: 1 if x == "MS" else 0).values,
    i=2,
    axes=axes.flatten(),
    type_data="singlecell",
    genes=cd4_csf.var_names,
    plot_title_suffix="(CD4 CSF)",
)

dependence_plot(
    df_expl=bcells_csf_expl,
    shap_object=bcells_csf_shapObject,
    data=bcells_csf.X,
    label_col=bcells_csf.obs["condition"].apply(lambda x: 1 if x == "MS" else 0).values,
    i=4,
    axes=axes.flatten(),
    type_data="singlecell",
    genes=bcells_csf.var_names,
    plot_title_suffix="(B-cells CSF)",
)

dependence_plot(
    df_expl=bcells_pbmc_expl,
    shap_object=bcells_pbmc_shapObject,
    data=bcells_pbmc.X,
    label_col=bcells_pbmc.obs["condition"].apply(lambda x: 1 if x == "MS" else 0).values,
    i=6,
    axes=axes.flatten(),
    type_data="singlecell",
    genes=bcells_pbmc.var_names,
    plot_title_suffix="(B-cells PBMC)",
)

plt.tight_layout()
plt.show()
# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()

# Display the entire figure with all plots
# In a Jupyter notebook, the last line of a cell returning a matplotlib figure
# will often display it automatically, but plt.show() is explicit and good practice.
plt.show()



# +
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd # Import pandas as it's used

# Modifica globale delle dimensioni dei font
plt.rc("axes", titlesize=18, labelsize=16)   # titoli assi e figure
plt.rc("legend", fontsize=14, title_fontsize=16)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
# Initialize your figure and axes


# Funzione per il dependence plot
def dependence_plot(
    df_expl,
    shap_object,
    data,
    label_col,
    ax, # Modified: Pass a single axes object
    feature_to_plot, # New: Specify which feature to plot (f1 or f2)
    type_data="microarray",
    genes=None,
    plot_title_suffix="",
):
    # Ensure genes is a list for consistent indexing later
    if genes is None and type_data == "singlecell":
        raise ValueError(
            "For 'singlecell' type_data, 'genes' parameter must be provided."
        )
    if genes is not None and not isinstance(genes, (list, pd.Index)):
        genes = list(genes)  # Convert to list if it's a different iterable

    # Handle different data types for 'data'
    if isinstance(data, pd.DataFrame):
        current = data.copy()
        feature_names = current.columns
    elif type_data == "singlecell":
        # For anndata.AnnData.X (often a numpy array or sparse matrix)
        # We assume 'genes' contains the feature names in the correct order
        current = pd.DataFrame(data, columns=genes)
        feature_names = genes
    else:
        raise TypeError(
            "Unsupported 'data' type. Must be pandas.DataFrame or compatible with 'singlecell' type_data."
        )

    # Note: MinMaxScaler is applied here, but for dependence plots, often raw data is preferred for feature values
    # However, if your SHAP values are based on scaled data, then it's consistent.
    # For this specific request, we keep it as it was in the original function.
    current_Scaled = MinMaxScaler().fit_transform(current)
    current_Scaled = pd.DataFrame(current_Scaled, columns=feature_names)

    top_features = df_expl.sort_values("MeanAbsoluteShap", ascending=False)[
        "Feature"
    ].values

    # Determine which feature to plot based on the argument
    if feature_to_plot == 'top1':
        feature_name = top_features[0]
    elif feature_to_plot == 'top2':
        feature_name = top_features[1]
    else:
        raise ValueError("feature_to_plot must be 'top1' or 'top2'")

    # Get index of feature for shap_object and original data
    idx_feature = feature_names.get_loc(feature_name)

    shap_vals = shap_object.values[:, idx_feature]
    feature_vals = current.iloc[:, idx_feature].values  # Use .iloc for consistent indexing

    customDf = pd.DataFrame(
        {
            "SHAP": shap_vals,
            "Feature": feature_vals,
            "Label": ["MS" if value == 1 else "Control" for value in label_col],
        }
    )

    sns.scatterplot(data=customDf, x="Feature", y="SHAP", hue="Label", palette="Set1", ax=ax)
    ax.set_title(
        f"Dependence plot for {feature_name} {plot_title_suffix}", fontsize=20
    )
    ax.set_xlabel(f"{feature_name}")
    ax.set_ylabel("SHAP value")
    ax.legend()


# Initialize your figure and axes for all plots
# 2 rows (for Top1 and Top2), and 4 columns (for each dataset)
fig, axes = plt.subplots(2, 4, figsize=(25, 12)) # Adjusted figsize for better visibility
# axes is now a 2D array: axes[row, col]

# Define your datasets in a list for easier iteration
datasets_info = [
    {
        "df_expl": microarray_explanations,
        "shap_object": microarray_shapObject,
        "data": microarray.drop(columns=["SampleID", "PatientID", "Label"]),
        "label_col": microarray["Label"].values,
        "type_data": "microarray",
        "genes": None,
        "plot_title_suffix": "(Microarray)",
    },
    {
        "df_expl": cd4_csf_expl,
        "shap_object": cd4_csf_shapObject,
        "data": cd4_csf.X,
        "label_col": cd4_csf.obs["condition"].apply(lambda x: 1 if x == "MS" else 0).values,
        "type_data": "singlecell",
        "genes": cd4_csf.var_names,
        "plot_title_suffix": "(CD4 CSF)",
    },
    {
        "df_expl": bcells_csf_expl,
        "shap_object": bcells_csf_shapObject,
        "data": bcells_csf.X,
        "label_col": bcells_csf.obs["condition"].apply(lambda x: 1 if x == "MS" else 0).values,
        "type_data": "singlecell",
        "genes": bcells_csf.var_names,
        "plot_title_suffix": "(BCells CSF)",
    },
    {
        "df_expl": bcells_pbmc_expl,
        "shap_object": bcells_pbmc_shapObject,
        "data": bcells_pbmc.X,
        "label_col": bcells_pbmc.obs["condition"].apply(lambda x: 1 if x == "MS" else 0).values,
        "type_data": "singlecell",
        "genes": bcells_pbmc.var_names,
        "plot_title_suffix": "(BCells PBMC)",
    },
]

# Plot Top 1 genes for each dataset in the first row
for col_idx, ds in enumerate(datasets_info):
    dependence_plot(
        df_expl=ds["df_expl"],
        shap_object=ds["shap_object"],
        data=ds["data"],
        label_col=ds["label_col"],
        ax=axes[0, col_idx], # First row (index 0)
        feature_to_plot='top1',
        type_data=ds["type_data"],
        genes=ds["genes"],
        plot_title_suffix=ds["plot_title_suffix"],
    )

# Plot Top 2 genes for each dataset in the second row
for col_idx, ds in enumerate(datasets_info):
    dependence_plot(
        df_expl=ds["df_expl"],
        shap_object=ds["shap_object"],
        data=ds["data"],
        label_col=ds["label_col"],
        ax=axes[1, col_idx], # Second row (index 1)
        feature_to_plot='top2',
        type_data=ds["type_data"],
        genes=ds["genes"],
        plot_title_suffix=ds["plot_title_suffix"],
    )

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()

# Display the entire figure with all plots
plt.show()
# +
def plot_samu(feature, dataset, shap_obj, subax, dataset_name):
    n_bins = 8
    index_feature = dataset.var_names.tolist().index(feature)
    
    # Handle AnnData X matrix (could be sparse)
    if hasattr(dataset.X, 'toarray'):
        feature_values = dataset.X[:, index_feature].toarray().flatten()
    else:
        feature_values = dataset.X[:, index_feature]
    
    # Create bins and format labels properly
    range_bins = pd.qcut(feature_values, q=n_bins, duplicates='drop')
    bins = pd.Series(range_bins).apply(lambda x: f"[{x.left:.3f}–{x.right:.3f}]")
    
    # Get SHAP values for this feature
    shap_values = shap_obj.values[:, index_feature]
    
    # Get condition labels - ensure same length
    conditions = dataset.obs.condition.values
    
    # Debug: print lengths to identify the issue
    print(f"Feature values length: {len(feature_values)}")
    print(f"Bins length: {len(bins)}")
    print(f"SHAP values length: {len(shap_values)}")
    print(f"Conditions length: {len(conditions)}")
    
    # Ensure all arrays have the same length
    min_length = min(len(bins), len(shap_values), len(conditions))
    
    df_plot = pd.DataFrame({
        'shap_value': shap_values[:min_length],
        'expression_bin': bins[:min_length],
        'condition': conditions[:min_length]
    })
    
    # Set colors for conditions
    palette = {'CTRL': 'lightblue', 'MS': 'lightcoral'}
    
    sns.boxplot(
        x='expression_bin',
        y='shap_value',
        hue='condition',
        data=df_plot,
        palette=palette,
        flierprops=dict(marker='o', color='black', markersize=4),
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black', linewidth=2),
        notch=True,
        ax=subax
    )
    
    subax.set_title(f'{feature} - {dataset_name}', fontsize=12, pad=10)
    subax.set_xlabel('Expression Bin', fontsize=10)
    subax.set_ylabel('SHAP Value', fontsize=10)
    subax.legend(title='Condition', fontsize=9)
    subax.tick_params(axis='x', rotation=45, labelsize=8)
    subax.tick_params(axis='y', labelsize=8)
import numpy as np

def microarray_samu_plot(subax, feature):
    # Load the data files
    microarray = pd.read_csv("./MICROARRAY/MergedDatasetFullCombatDeclustered_symbol.csv")
    microarray_shapObject = joblib.load("./MICROARRAY/xgbDefFull_shapValues.pkl")
    map_microarray = pd.read_csv("./MICROARRAY/patientMap.csv")

    # Rename the 'Label' column from the map file to 'stage'
    map_microarray.rename(columns={"Label": "stage"}, inplace=True)

    # Merge the stage information from the map into the main dataframe
    microarray_merged = pd.merge(
        microarray,
        map_microarray[["SampleID", "stage"]],
        on="SampleID",
        how="left",
    )

    # Create a single, clean 'condition' column for plotting.
    microarray_merged["condition"] = np.where(
        microarray_merged["Label"] == 0,
        "Control",
        microarray_merged["stage"].fillna("MS"),
    )
    labels = microarray_merged["condition"]

    # Prepare Data for Plotting
    # Drop metadata columns to create the feature matrix.
    microarray.drop(columns=["SampleID", "PatientID", "Label"], inplace=True)

    n_bins = 8
    range_bins = pd.qcut(microarray[feature], q=n_bins, duplicates="drop")
    bins = pd.Series(range_bins).apply(lambda x: f"[{x.left:.1f}–{x.right:.1f}]")

    # Get SHAP values for the specific feature
    shap_values = microarray_shapObject.values[
        :, microarray.columns.get_loc(feature)
    ]

    # Create the final DataFrame for seaborn
    df_plot = pd.DataFrame(
        {
            "shap_value": shap_values,
            "expression_bin": bins,
            "condition": labels,
        }
    )

    # Define the color palette for all expected conditions
    palette = {
        "Control": "lightblue",
        "MS": "lightcoral",
        "PP": "mediumseagreen",
        "RR": "mediumpurple",
        "SP": "gold",
        "CIS": "darkorange",
    }

    # --- FIX IS HERE: Define the desired order for the plot categories ---
    hue_order = ["Control", "CIS", "RR", "SP", "PP", "MS"]

    # Generate the boxplot
    sns.boxplot(
        x="expression_bin",
        y="shap_value",
        hue="condition",
        hue_order=hue_order,  # <-- ADD THE HUE_ORDER PARAMETER
        data=df_plot,
        palette=palette,
        flierprops=dict(marker="o", color="black", markersize=4),
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black", linewidth=2),
        notch=True,
        ax=subax,
    )

    subax.set_title("ABCA1 - MICROARRAY", fontsize=12, pad=10)
    subax.set_xlabel("Expression Bin", fontsize=10)
    subax.set_ylabel("SHAP Value", fontsize=10)
    subax.legend(title="Condition", fontsize=9)
    subax.tick_params(axis="x", rotation=45, labelsize=8)
    subax.tick_params(axis="y", labelsize=8)

fig, axes = plt.subplots(2, 4, figsize=(48, 24))

# Flatten axes for easier indexing
axes = axes.flatten()

# Now you can just use axes[0] ... axes[7]
microarray_samu_plot(axes[0], "ABCA1")
microarray_samu_plot(axes[1], "NDUFS5")

plot_samu('ALDOA', cd4_csf, cd4_csf_shapObject, axes[2], "CD4_CSF")
plot_samu('MTRNR2L1', cd4_csf, cd4_csf_shapObject, axes[3], "CD4_CSF")

plot_samu('CD300LD', bcells_csf, bcells_csf_shapObject, axes[4], "BCELLS_CSF")
plot_samu('LRRC8E', bcells_csf, bcells_csf_shapObject, axes[5], "BCELLS_CSF")

plot_samu("SLC30A4", bcells_pbmc, bcells_pbmc_shapObject, axes[6], "BCELLS_PBMC")
plot_samu("APBB2", bcells_pbmc, bcells_pbmc_shapObject, axes[7], "BCELLS_PBMC")

# Improve spacing and layout
plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
fig.suptitle("SHAP Values by Expression Bin and Condition Across Datasets", 
             fontsize=16, y=0.98)

plt.show()


# -

def plot_samu(feature, dataset, shap_obj, subax, dataset_name):
    n_bins = 8
    index_feature = dataset.var_names.tolist().index(feature)
    range_bins = pd.qcut(dataset.X[:, index_feature], q=n_bins)
    bins = range_bins.astype(str)
    df_plot = pd.DataFrame({
        'shap_value': shap_obj.values[:, index_feature],
        'expression_bin': bins,
        'condition': dataset.obs.condition
    })
    
    # Imposta i colori per le condizioni
    palette = {'CTRL': 'lightblue', 'MS': 'lightcoral'}
    plt.figure(figsize=(8, 8))
    # Use the 'ax' parameter to plot on the provided subplot
    sns.boxplot(
        x='expression_bin',
        y='shap_value',
        hue='condition',
        data=df_plot,
        palette=palette,
        flierprops=dict(marker='o', color='black', markersize=6),
        boxprops=dict(edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black', linewidth=2),
        notch=True,
        ax=subax  # Key change: pass the subplot object here
    )
    # Use the subplot object (subax) for setting titles, labels, etc.
    subax.set_title(f'SHAP values for {feature} by expression bin and condition (DATASET: {dataset_name})', fontsize=15)
    subax.set_xlabel('Expression Bin')
    subax.set_ylabel('SHAP Value')
    subax.legend(title='Condition')
    subax.tick_params(axis='x', rotation=45) 


# +
# --- Create a single, larger plot ---

# 1. Create a figure with a single subplot area (ax)
fig, ax = plt.subplots(2,1, figsize=(12, 16))

# 2. Call your plotting function, passing the single axis 'ax'
microarray_samu_plot(ax[0], "ABCA1")
microarray_samu_plot(ax[1], "NDUFS5")

# 3. Adjust layout to prevent labels from being cut off
plt.tight_layout()

# 4. Display the plot
plt.show()
# -


cd4_csf.X[:,200]


def microarray_samu_plot(subax, gene):
    # Load the data files
    microarray = pd.read_csv("./MICROARRAY/MergedDatasetFullCombatDeclustered_symbol.csv")
    microarray_shapObject = joblib.load("./MICROARRAY/xgbDefFull_shapValues.pkl")
    map_microarray = pd.read_csv("./MICROARRAY/patientMap.csv")

    # Rename the 'Label' column from the map file to 'stage'
    map_microarray.rename(columns={"Label": "stage"}, inplace=True)

    # Merge the stage information from the map into the main dataframe
    microarray_merged = pd.merge(
        microarray,
        map_microarray[["SampleID", "stage"]],
        on="SampleID",
        how="left",
    )

    # Create a single, clean 'condition' column for plotting.
    microarray_merged["condition"] = np.where(
        microarray_merged["Label"] == 0,
        "Control",
        microarray_merged["stage"].fillna("MS"),
    )
    labels = microarray_merged["condition"]

    # Prepare Data for Plotting
    # Drop metadata columns to create the feature matrix.
    microarray.drop(columns=["SampleID", "PatientID", "Label"], inplace=True)

    feature = gene
    n_bins = 8
    range_bins = pd.qcut(microarray[feature], q=n_bins, duplicates="drop")
    bins = pd.Series(range_bins).apply(lambda x: f"[{x.left:.1f}–{x.right:.1f}]")

    # Get SHAP values for the specific feature
    shap_values = microarray_shapObject.values[
        :, microarray.columns.get_loc(feature)
    ]

    # Create the final DataFrame for seaborn
    df_plot = pd.DataFrame(
        {
            "shap_value": shap_values,
            "expression_bin": bins,
            "condition": labels,
        }
    )

    # Define the color palette for all expected conditions
    palette = {
        "Control": "lightblue",
        "MS": "lightcoral",
        "PP": "mediumseagreen",
        "RR": "mediumpurple",
        "SP": "gold",
        "CIS": "darkorange",
    }

    # --- FIX IS HERE: Define the desired order for the plot categories ---
    hue_order = ["Control", "CIS", "RR", "SP", "PP", "MS"]

    # Generate the boxplot
    sns.boxplot(
        x="expression_bin",
        y="shap_value",
        hue="condition",
        hue_order=hue_order,  # <-- ADD THE HUE_ORDER PARAMETER
        data=df_plot,
        palette=palette,
        flierprops=dict(marker="o", color="black", markersize=4),
        boxprops=dict(edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black", linewidth=2),
        notch=True,
        ax=subax,
    )

    subax.set_title(f"{gene} - MICROARRAY", fontsize=12, pad=10)
    subax.set_xlabel("Expression Bin", fontsize=10)
    subax.set_ylabel("SHAP Value", fontsize=10)
    subax.legend(title="Condition", fontsize=9)
    subax.tick_params(axis="x", rotation=45, labelsize=8)
    subax.tick_params(axis="y", labelsize=8)



