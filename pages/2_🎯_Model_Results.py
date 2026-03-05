import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, root_mean_squared_error
import os

st.set_page_config(page_title="Model Results", page_icon="🎯", layout="wide")


@st.cache_data
def load_data():
    try:
        # Try loading your actual files
        df = pd.read_csv('data/AirBNB.csv')
        y_df = pd.read_csv('data/xgb_predictions_aligned.csv')
        y_actual = y_df['actual'].values
        y_pred = y_df['prediction'].values
    except FileNotFoundError:
        # Create sample data if files are missing so the script doesn't crash
        st.warning("Data files not found. Displaying synthetic sample data.")
        np.random.seed(42)
        y_actual = np.random.exponential(scale=150, size=5000) + 50
        # Simulate a model with some noise and bias
        y_pred = y_actual * 0.85 + np.random.normal(0, 40, size=5000)

    return y_actual, y_pred


y_actual, y_pred = load_data()

# --- 3. OPTIMIZED CALCULATIONS (CACHED) ---
@st.cache_data
def get_plot_metrics(y_actual, y_pred):
    # Downsample for performance (Plotly struggles with >5k points in SVG)
    if len(y_actual) > 5000:
        indices = np.random.choice(len(y_actual), 5000, replace=False)
        y_a_sub, y_p_sub = y_actual[indices], y_pred[indices]
    else:
        y_a_sub, y_p_sub = y_actual, y_pred

    # Calculate Regression Line
    z = np.polyfit(y_a_sub, y_p_sub, 1)
    p = np.poly1d(z)
    sort_idx = np.argsort(y_a_sub)

    return y_a_sub, y_p_sub, y_a_sub[sort_idx], p(y_a_sub[sort_idx])


y_sub_a, y_sub_p, x_fitted, y_fitted = get_plot_metrics(y_actual, y_pred)

# Global Metrics
r2 = r2_score(y_actual, y_pred)
rmse = root_mean_squared_error(y_actual, y_pred)
mae = np.mean(np.abs(y_actual - y_pred))

print(f"**********: {rmse}")

st.title("🎯 Model Training Results")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Model Comparison")
    comparison_df = pd.DataFrame({
        'Model': ['XGBoost (Final)', 'OLS', 'Random Forest'],
        'RMSE': [round(rmse, 2), 114.75, 0.0],
        'R²': [round(r2, 3), 0.55, 0.0],
    })
    st.dataframe(comparison_df, width="stretch")

with col2:
    fig_comp = px.bar(comparison_df, x='Model', y='RMSE', color='Model',
                      title='Model RMSE Comparison (Lower is Better)')
    st.plotly_chart(fig_comp, width="stretch")

st.divider()


st.header("Feature Selection")
if os.path.exists("assets/feature_selection.png"):
    st.image("assets/feature_selection.png", width="stretch")
else:
    st.info("Place 'feature_selection.png' in assets folder to display.")

st.image("assets/training.png", width="stretch")


st.header("Visualizing Prediction Accuracy")

# Initialize Figure
fig = go.Figure()

# Scatter Points using Scattergl (GPU Accelerated)
fig.add_trace(go.Scattergl(
    x=y_sub_a, y=y_sub_p,
    mode='markers',
    name='Sampled Listings',
    marker=dict(opacity=0.5, color='royalblue', size=6)
))

# Fitted Line
fig.add_trace(go.Scatter(
    x=x_fitted, y=y_fitted,
    mode='lines',
    name='Fitted Trend',
    line=dict(color='red', width=2)
))

# Perfect Prediction Identity Line
max_val = max(y_actual.max(), y_pred.max())
fig.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val],
    mode='lines',
    name='Ideal (Perfect)',
    line=dict(color='green', dash='dash')
))

fig.update_layout(
    title=dict(
        text=f"Actual vs Predicted Prices<br><span style='font-size: 0.8em; color: gray;'>Calculated on Full Set: R² = {r2:.4f}, RMSE = ${rmse:.2f}</span>",
    ),
    xaxis_title="Actual Price ($)",
    yaxis_title="Predicted Price ($)",
    template="plotly_white",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, width="stretch")

st.header("Performance by Price Range")
if os.path.exists("assets/price_range_pred.png"):
    st.image("assets/price_range_pred.png", width="stretch")
else:
    st.info("Place 'price_range_pred.png' in assets folder to display.")

st.header("Error Analysis (Residuals)")
residuals = y_actual - y_pred

fig_res = px.scatter(
    x=y_pred, y=residuals,
    render_mode='webgl',  # Use WebGL for speed
    labels={'x': 'Predicted Price ($)', 'y': 'Residual ($)'},
    opacity=0.4,
    title='Residual Plot: errors increase with price',
    template="plotly_white",
    color_discrete_sequence=['purple']
)
fig_res.add_hline(y=0, line_dash="dash", line_color="red")

st.plotly_chart(fig_res, width="stretch")