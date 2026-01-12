# main.py
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from config import (feature_bounds, default_values, feature_steps, 
                    rename_dict, input_cols, input_cols_fe)
from helpers.feature_utils import compute_derived_features
from helpers.input_utils import make_synced_input
from helpers.model_utils import load_model, extract_model_params, get_model_names
import shap
import warnings
warnings.filterwarnings(
    "ignore",
    message="The widget with key .* was created with a default value"
)


from ensemble import L0BaggingModelRegressor, HybridChainPredictor

# ------------------ App Title ------------------
st.set_page_config(
    page_title="SCM+RAC Concrete Predictor",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

st.title("SCM+RAC Concrete Prediction")

raw_features = list(feature_bounds.keys())
input_data = {}

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=raw_features + ['Prediction'])

# ------------------ Prediction ------------------
# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.current_model_info = None

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


# ------------------ Sidebar Model Selection ------------------
st.sidebar.header("Prediction")

# Load model
with st.sidebar:
    model = load_model()
    st.session_state.model_loaded = True
    st.session_state.model = model

    # Predict button
    predict_disabled = not st.session_state.get("model_loaded", False)
    if st.button("Predict"):
        start_time = time.time()
        with st.spinner("Predicting..."):
            predictions_dict = st.session_state.model.predict(X_input)
        elapsed = time.time() - start_time
        
        # Display predictions as key-value pairs
        st.sidebar.markdown(f"**Predictions:**")
        for output_name, pred_values in predictions_dict.items():
            if isinstance(pred_values, np.ndarray):
                pred_value = pred_values[0] if pred_values.size > 0 else 0
            else:
                pred_value = pred_values
            st.sidebar.metric(label=output_name, value=f"{pred_value:.2f} kN")
        
        st.sidebar.markdown(f"**Time:** {elapsed:.2f} seconds")

        # Update prediction history
        new_row = input_data.copy()
        for output_name, pred_values in predictions_dict.items():
            if isinstance(pred_values, np.ndarray):
                new_row[output_name] = pred_values[0] if pred_values.size > 0 else 0
            else:
                new_row[output_name] = pred_values
        
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([new_row])],
            ignore_index=True
        )

# ------------------ SHAP Analysis ------------------
# with tabs[1]:
#     if not st.session_state.get("model_loaded", False):
#         st.warning("Load a model in the Prediction tab first.")
        

# ------------------ Scenario Analysis ------------------
# with tabs[2]:
    # if not st.session_state.get("model_loaded", False):
    #     st.warning("Load a model in the Prediction tab first.")
    # else:
    #     feature_to_vary = st.selectbox("Select feature to vary", options=raw_features)
    #     st.markdown("---")
    #     lb, ub = feature_bounds[feature_to_vary]
    #     vary_range = st.slider(f"Set range for {rename_dict[feature_to_vary]}", 
    #                            min_value=lb, max_value=ub, value=(lb, ub))
    #     n_points = st.slider("Number of points in scenario", min_value=5, max_value=100, 
    #                          value=20, step=5)
    #     vary_values = np.linspace(vary_range[0], vary_range[1], n_points)

    #     scenario_df = pd.DataFrame([input_data]*n_points)
    #     scenario_df[feature_to_vary] = vary_values
    #     scenario_df = compute_derived_features(scenario_df)

    #     cols_to_drop = ['rho_h', 'rho_v', 'rho_b', 'fyh', 'fyv', 'fyb']
    #     X_scenario = scenario_df.drop(columns=[col for col in cols_to_drop 
    #                                            if col in scenario_df.columns])
    #     predictions = st.session_state.model.predict(X_scenario.values)
        
        # fig = px.line(x=vary_values, y=predictions)
        # fig.update_layout(xaxis_title=f"{feature_to_vary} [mm]", 
        #                   yaxis_title="Predicted Shear Strength (kN)")
        # st.plotly_chart(fig, use_container_width=True)

# ------------------ Prediction History ------------------
# with tabs[3]:
    # history = st.session_state.history

    # if history.empty:
    #     st.warning("No prediction history available.")
    # else:
    #     if st.button("Clear Prediction History"):
    #         st.session_state.history = pd.DataFrame(columns=raw_features + ['Prediction'])
    #         history = st.session_state.history

    #     st.dataframe(history)
    #     st.markdown("---")
    #     # Feature vs Prediction
    #     feature_to_plot = st.selectbox("Select feature to plot", options=raw_features)
    #     fig_feature = px.scatter(history, x=feature_to_plot, y="Prediction", 
    #                              hover_data=history.columns,
    #                              labels={feature_to_plot: feature_to_plot, 
    #                                      "Prediction": "Shear Strength (kN)"},
    #                              title=f"{feature_to_plot} vs Predicted Shear Strength")
    #     st.plotly_chart(fig_feature, use_container_width=True)
        
    #     st.markdown("---")
    #     # Historical Shear Strength Plot
    #     fig3 = px.scatter(history, x=history.index, y="Prediction", hover_data=history.columns,
    #                       labels={"x": "Prediction Index", "Prediction": "Shear Strength (kN)"},
    #                       title="Prediction History")
    #     st.plotly_chart(fig3, use_container_width=True)
