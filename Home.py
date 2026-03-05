import streamlit as st
import requests
import plotly.graph_objects as go
import json

# Page config
st.set_page_config(
    page_title="AirBnB Price Predictor",
    page_icon="assets/bnb.jpg",
    layout="wide"
)

API_URL = "http://localhost:8000"

st.title("AirBnB Price Predictor")
st.markdown("Get instant price predictions for your AirBnB listing")

# Add model selector at the top after header
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    model_choice = st.radio(
        "🤖 Select Prediction Model",
        ["OLS Regression", "XGBoost"],
        horizontal=True,
        help="Compare predictions from different models"
    )

    # Show model info
    if model_choice == "OLS Regression":
        st.info(" **OLS**: Linear model | RMSE: $114.75 | R²: 0.54 | Fast predictions")
    else:
        st.info(" **XGBoost**: Tree-based ensemble | RMSE: $101.50 | R²: 0.60 | Higher accuracy")

st.markdown("---")


# Sidebar for model info
with st.sidebar:
    st.header("Model Info")
    st.metric("Model", "XGBoost")
    st.metric("Test RMSE", "$89.50")  # Your actual result
    st.metric("R² Score", "0.68")  # Your actual result
    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("- XGBoost Regressor")
    st.markdown("- FastAPI Backend")
    st.markdown("- Streamlit Frontend")

# Main prediction form
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Location")
    city = st.selectbox("City", ["LA", "SF", "NYC", "Chicago", "Boston", "DC"])

    st.subheader("Property Details")
    property_type = st.selectbox("Property Type", ["Apartment", "House", "Other"])
    room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])

with col2:
    st.subheader("Capacity")
    accommodates = st.slider("Guests", 1, 20, 4)
    bedrooms = st.slider("Bedrooms", 0, 20, 2)
    bathrooms = st.slider("Bathrooms", 0.0, 20.0, 1.5, 0.5)
    beds = st.slider("Beds", 1, 20, 2)

with col3:
    st.subheader("Pricing & Policies")
    cleaning_fee = st.number_input("Cleaning Fee ($)", 0, 500, 10)
    cancellation = st.selectbox("Cancellation Policy",
                                ["flexible", "moderate", "strict", "super_strict_60"])

    with st.expander("⚙ Advanced Options"):
        review_score = st.slider("Review Score (0-100)", 0, 100, 0)
        num_reviews = st.number_input("Number of Reviews", 0, 500, 0)
        host_response_rate = st.slider("Host Response Rate (%)", 0, 100, 0)
        host_verified = st.checkbox("Host Identity Verified", value=True)

# Predict button
if st.button("Predict Price", type="primary", use_container_width=True):
    with st.spinner("Calculating price..."):
        # Prepare request
        payload = {
            "city": city,
            "accommodates": int(accommodates),
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms),
            "beds": int(beds),
            "room_type": room_type,
            "property_type": property_type,
            "cancellation_policy": cancellation,
            "cleaning_fee": float(cleaning_fee),
            "review_scores_rating": float(review_score),
            "number_of_reviews": int(num_reviews),
            "host_response_rate": float(host_response_rate),
            "host_identity_verified": bool(host_verified),
            "model_type": "xgboost" if model_choice == "XGBoost" else "ols"  # ADD THIS
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.success("✅ Prediction Complete!")


            # Display which model was used
            st.markdown(f"## 🎉 Prediction Results ({result['model']})")

            # Display results

            # Main price display
            col1, col2, col3 = st.columns([2, 1, 1])

            # Model comparison
            with st.expander("🔬 Compare Both Models"):
                col1, col2 = st.columns(2)

                # Get predictions from both models
                payload_ols = {**payload, "model_type": "ols"}
                payload_xgb = {**payload, "model_type": "xgboost"}

                try:
                    result_ols = requests.post(f"{API_URL}/predict", json=payload_ols, timeout=10).json()
                    result_xgb = requests.post(f"{API_URL}/predict", json=payload_xgb, timeout=10).json()

                    with col1:
                        st.markdown("### OLS Regression")
                        st.metric("Predicted Price", f"${result_ols['predicted_price']:.2f}")
                        st.caption(f"RMSE: $119.16 | R²: 0.51")

                    with col2:
                        st.markdown("### XGBoost")
                        st.metric(
                            "Predicted Price",
                            f"${result_xgb['predicted_price']:.2f}",
                            delta=f"${result_xgb['predicted_price'] - result_ols['predicted_price']:.2f}"
                        )
                        st.caption(f"RMSE: $89.50 | R²: 0.68")

                    # Show difference
                    diff = abs(result_xgb['predicted_price'] - result_ols['predicted_price'])
                    st.info(
                        f"💡 **Price difference between models:** ${diff:.2f} ({(diff / result_ols['predicted_price'] * 100):.1f}%)")

                except Exception as e:
                    st.error(f"Error comparing models: {str(e)}")

            with col1:
                st.metric(
                    "Predicted Nightly Price",
                    f"${result['predicted_price']:.2f}",
                    delta=f"{result['city_comparison']['difference_pct']:.1f}% vs city avg"
                )

            with col2:
                st.metric(
                    "Lower Bound (95% CI)",
                    f"${result['confidence_interval']['lower']:.2f}"
                )

            with col3:
                st.metric(
                    "Upper Bound (95% CI)",
                    f"${result['confidence_interval']['upper']:.2f}"
                )

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Price range visualization
                fig = go.Figure()

                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['predicted_price'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Price"},
                    delta={'reference': result['city_comparison']['city_average']},
                    gauge={
                        'axis': {'range': [None, result['confidence_interval']['upper'] * 1.2]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, result['confidence_interval']['lower']], 'color': "lightgray"},
                            {'range': [result['confidence_interval']['lower'],
                                       result['confidence_interval']['upper']], 'color': "lightblue"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': result['city_comparison']['city_average']
                        }
                    }
                ))

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Top features
                st.subheader("Top Price Drivers")
                features = result['top_features']

                fig = go.Figure(go.Bar(
                    x=[f['importance'] for f in features],
                    y=[f['feature'] for f in features],
                    orientation='h',
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.subheader("Pricing Insights")

            diff_pct = result['city_comparison']['difference_pct']
            if diff_pct > 20:
                st.info(
                    f"Your listing is priced **{diff_pct:.1f}% above** the {city} average. Consider if your amenities justify the premium.")
            elif diff_pct < -20:
                st.warning(
                    f"Your listing is priced **{abs(diff_pct):.1f}% below** the {city} average. You may be underpricing!")
            else:
                st.success(f"✅ Your pricing is competitive, within **{abs(diff_pct):.1f}%** of the {city} average.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to prediction service: {str(e)}")
            st.info("Make sure the FastAPI backend is running on http://localhost:8000")


# Quick tips
with st.expander("Tips for Better Pricing"):
    st.markdown("""
    - **Location matters**: LA and DC typically command higher average prices
    - **Entire home** listings price 40-60% higher than private rooms
    - **Reviews count**: Properties with 20+ reviews get 15% premium
    - **Cleaning fee**: Keep under $100 for better booking rates
    - **Strict cancellation**: May reduce bookings by 10-15%
    """)