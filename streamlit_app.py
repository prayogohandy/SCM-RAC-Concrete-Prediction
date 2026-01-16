# main.py
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from config import (feature_bounds, default_values, feature_steps, rename_input, reverse_rename_input,
                    rename_dict, reverse_rename_dict, input_cols_fe, input_cols_all, OUTPUT_FORMAT)
from helpers.feature_utils import compute_derived_features
from helpers.input_utils import make_synced_input
from helpers.model_utils import load_model, load_shap_explainer
import shap
import warnings
warnings.filterwarnings(
    "ignore",
    message="The widget with key .* was created with a default value"
)


from ensemble import L0BaggingModelRegressor, HybridChainPredictor, HybridChainSHAPExplainer

app_title = "Cross-Output Information Transfer - Concrete Properties Predictor"

# ------------------ App Title ------------------
st.set_page_config(
    page_title=app_title,
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

st.title(app_title)

raw_features = list(feature_bounds.keys())
input_data = {}

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=raw_features)

# ------------------ Tabs ------------------
tabs = st.tabs(["Concrete Details", "SHAP Analysis", "Scenario Analysis", "History"])

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Input Parameters")
with st.sidebar.expander("Recycled Aggregate Concrete", expanded=False):
    container = st.container()
    # Main Geometry Inputs
    rac_inputs = ['C', 'W', 'CA', 'RCA', 'FAg', 'SP']
    for i, f in enumerate(rac_inputs):
        lb, ub = feature_bounds[f]
        input_data[f] = make_synced_input(f, lb, ub, feature_steps[f], container, rename_dict, 
                                          default_values=default_values)
        
with st.sidebar.expander("Supplementary Cementitious Materials", expanded=False):
    # Section Properties Inputs
    container = st.container()
    scm_inputs = ['FA', 'SF', 'GGBFS', 'CC']
    for i, f in enumerate(scm_inputs):
        lb, ub = feature_bounds[f]
        scale = 100 if f.startswith('rho') else 1
        input_data[f] = make_synced_input(f, lb, ub, feature_steps[f], container, rename_dict, 
                                          scale=scale, default_values=default_values)

column_order = ['C', 'W', 'CA', 'RCA', 'FAg', 'FA', 'SF', 'GGBFS', 'CC', 'SP']
input_data = {k: input_data[k] for k in column_order if k in input_data}
st.sidebar.markdown("---")

# ------------------ Inputs & Prediction ------------------
with tabs[0]:
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    # ------------------ Derived Features ------------------
    st.subheader("Derived Features")
    df = compute_derived_features(df)
    derived_cols = input_cols_fe
    columns = st.columns(3)
    for i, col_name in enumerate(derived_cols):
        format_str = "{:.2f}"
        value = df[col_name].values[0]
        if col_name in ['Cement%', 
            'FA%', 'SF%', 'GGBFS%', 'CC%', 
            'Coarse%',]:
            value = 100 * value
            format_str = "{:.1f}%"
        columns[i % 3].metric(label=rename_dict[col_name], value=format_str.format(value))

    X_input = df.values
    st.markdown("---")

    st.subheader("Concrete Mix")

    components = ['C', 'FA', 'SF', 'GGBFS', 'CC', 'CA', 'RCA', 'FAg', 'SP', 'W']
    values = [input_data[f] for f in components]

    fig = px.pie(
        names=[rename_input[f].replace(" (kg/m³)", "") for f in components],
        values=values,
        title="Concrete Composition by Weight",
        color=components
    )
    st.plotly_chart(fig)
    
    columns = st.columns(2)
    binder_components = ['C', 'FA', 'SF', 'GGBFS', 'CC']
    binder_values = [input_data[f] for f in binder_components]

    fig = px.pie(
        names=[rename_input[f].replace(" (kg/m³)", "") for f in binder_components],
        values=binder_values,
        title="Binder Composition by Weight",
        color=binder_components
    )
    columns[0].plotly_chart(fig, use_container_width=True)
    agg_components = ['CA', 'RCA', 'FAg']
    agg_values = [input_data[f] for f in agg_components]

    fig = px.pie(
        names=[rename_input[f].replace(" (kg/m³)", "") for f in agg_components],
        values=agg_values,
        title="Aggregate Composition by Weight",
        color=agg_components
    )
    columns[1].plotly_chart(fig, use_container_width=True)

# ------------------ Sidebar Model Selection ------------------
st.sidebar.header("Prediction")

# Load model
with st.sidebar:
    model = load_model()
    st.session_state.model = model

    # Predict button
    if st.button("Predict"):
        start_time = time.time()
        with st.spinner("Predicting..."):
            predictions_dict = st.session_state.model.predict(X_input)
        elapsed = time.time() - start_time
        
        # Display predictions as key-value pairs
        rows = []
        new_row = input_data.copy()
        for output_name, pred_values in predictions_dict.items():
            value = pred_values[0] if isinstance(pred_values, np.ndarray) else pred_values
            unit = OUTPUT_FORMAT[output_name]['unit']
            decimal = OUTPUT_FORMAT[output_name]['decimals']
            rows.append({
                "Property": output_name,
                "Prediction": round(value, 2),
                "Unit": OUTPUT_FORMAT[output_name]["unit"],
                "Value": f"{value:.{decimal}f} {unit}"
            })

            # Update prediction history
            new_row[output_name] = value
            
        df_pred = pd.DataFrame(rows)
        st.sidebar.table(df_pred[["Property", "Value"]])
        st.sidebar.markdown(f"**Time:** {elapsed:.2f} seconds")

        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([new_row])],
            ignore_index=True
        )

# ------------------ SHAP Analysis ------------------
with tabs[1]:
    model = st.session_state.model
    shap_explainer = load_shap_explainer()

    output_options = [rename_dict[i] for i in OUTPUT_FORMAT.keys()]
    selected_display  = st.selectbox("Select output", options=output_options)
    output_selection = reverse_rename_dict[selected_display]
    shap_values, X_bg, feature_names = shap_explainer.get_global_shap(output_selection)
    max_display = st.slider("Max features to display", min_value=5, max_value=len(feature_names), value=10, step=1)
    
    # Global Explaination based on the model
    st.subheader("Global Feature Importance")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_bg, feature_names=feature_names, max_display=max_display, show=False)
    st.pyplot(fig, bbox_inches='tight')

    # Local Explaination based on the feature
    st.subheader("Local Feature Importance")
    explainations = shap_explainer.explain_local_all(X_input)
    fig, ax = plt.subplots()
    shap.plots.waterfall(explainations[output_selection][0], max_display=max_display, show=False)
    st.pyplot(fig, bbox_inches='tight')
        

# ------------------ Scenario Analysis ------------------
with tabs[2]:
    input_options = [rename_input[i] for i in raw_features]
    selected_input = st.selectbox("Select feature to vary", options=input_options)
    feature_to_vary = reverse_rename_input[selected_input]
    st.markdown("---")
    lb, ub = feature_bounds[feature_to_vary]
    vary_range = st.slider(f"Set range for {rename_dict[feature_to_vary]}", 
                            min_value=lb, max_value=ub, value=(lb, ub))
    n_points = st.slider("Number of points in scenario", min_value=5, max_value=100, 
                            value=20, step=5)
    vary_values = np.linspace(vary_range[0], vary_range[1], n_points)

    scenario_df = pd.DataFrame([input_data]*n_points)
    scenario_df[feature_to_vary] = vary_values
    scenario_df = compute_derived_features(scenario_df)

    predictions_dict = st.session_state.model.predict(scenario_df.values)

    output_options = [rename_dict[i] for i in OUTPUT_FORMAT.keys()]
    selected_display  = st.selectbox("Select output", options=output_options, key="scenario_output_select")
    output_selection = reverse_rename_dict[selected_display]
    fig = px.line(x=vary_values, y=predictions_dict[output_selection])
    fig.update_layout(xaxis_title=f"Input {rename_input[feature_to_vary]}", 
                        yaxis_title=f"Predicted {output_selection} ({OUTPUT_FORMAT[output_selection]['unit']})")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Prediction History ------------------
with tabs[3]:
    history = st.session_state.history

    if history.empty:
        st.warning("No prediction history available.")
    else:
        if st.button("Clear Prediction History"):
            st.session_state.history = pd.DataFrame(columns=raw_features + list(OUTPUT_FORMAT.keys()))
            history = st.session_state.history

        st.dataframe(history, use_container_width=True)
        st.markdown("---")

        # Select feature
        display_feature = st.selectbox(
            "Select feature to plot",
            options=input_options
        )
        feature_to_plot = reverse_rename_input[display_feature]
        # Select output

        display_output = st.selectbox(
            "Select output to plot",
            options=output_options
        )
        output_to_plot = reverse_rename_dict[display_output]

        unit = OUTPUT_FORMAT[output_to_plot]["unit"]

        fig_feature = px.scatter(
            history,
            x=feature_to_plot,
            y=output_to_plot,
            hover_data=history.columns,
            labels={
                feature_to_plot: rename_input[feature_to_plot],
                output_to_plot: f"{output_to_plot} ({unit})"
            },
            title=f"{feature_to_plot} vs {output_to_plot} Plot"
        )
        st.plotly_chart(fig_feature, use_container_width=True)

        st.markdown("---")

        fig_history = px.scatter(
            history,
            x=history.index,
            y=output_to_plot,
            hover_data=history.columns,
            labels={
                "x": "Prediction Index",
                output_to_plot: f"{output_to_plot} ({unit})"
            },
            title=f"{output_to_plot} Prediction History"
        )
        st.plotly_chart(fig_history, use_container_width=True)
