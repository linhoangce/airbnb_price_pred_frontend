import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EDA", page_icon="assets/eda.png", layout="wide")


# --- 2. DATA LOADING ---
# We use st.cache_data so the app stays fast even if the CSV is large
@st.cache_data
def load_data():
    # Relative path: works on local and Streamlit Cloud
    df = pd.read_csv("data/AirBNB.csv")

    property_mapping = {
        'Apartment': 'Apartment',
        'Condominium': 'Apartment',
        'Loft': 'Apartment',

        # Houses
        'House': 'House',
        'Townhouse': 'House',
        'Villa': 'House',
        'Bungalow': 'House',
        'In-law': 'House',
        'Cabin': 'House',
        'Chalet': 'House',

        # Guest Accommodations
        'Guesthouse': 'Guest_Accommodation',
        'Guest suite': 'Guest_Accommodation',
        'Bed & Breakfast': 'Guest_Accommodation',

        # Hotels/Hostels
        'Hostel': 'Hotel_Style',
        'Boutique hotel': 'Hotel_Style',
        'Serviced apartment': 'Hotel_Style',

        # Unique/Specialty
        'Boat': 'Unique',
        'Camper/RV': 'Unique',
        'Treehouse': 'Unique',
        'Yurt': 'Unique',
        'Castle': 'Unique',
        'Cave': 'Unique',
        'Tent': 'Unique',
        'Tipi': 'Unique',
        'Hut': 'Unique',
        'Island': 'Unique',
        'Train': 'Unique',
        'Earth House': 'Unique',

        # Other/Miscellaneous
        'Other': 'Other',
        'Dorm': 'Other',
        'Timeshare': 'Other',
        'Vacation home': 'Other',
        'Casa particular': 'Other',
        'Parking Space': 'Other'
    }
    df['property_type_grouped'] = df['property_type'].map(property_mapping).fillna('Other')

    return df


df = load_data()

# --- 3. TITLE & HEADER ---
st.title("📊 Exploratory Data Analysis")

# --- 4. VISUALIZATIONS ---

# Row 1: Price Distribution
st.header("Price Distribution by City")
fig_price = px.box(
    df,
    x="city",
    y="price",
    color="city",
    points="outliers",
    title="Price Spread across Cities",
    template="plotly_white"
)
st.plotly_chart(fig_price, use_container_width=True, key="eda_price_box")

# Row 2: Neighborhoods
st.header("Neighborhood Pricing Analysis")
top_neigh = df.groupby('neighbourhood')['price'].median().sort_values(ascending=False).head(15).reset_index()

fig_neigh = px.bar(
    top_neigh,
    x='price',
    y='neighbourhood',
    orientation='h',
    color='price',
    color_continuous_scale='Viridis',
    title="Top 15 Most Expensive Neighborhoods (Median)"
)
fig_neigh.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_neigh, use_container_width=True, key="eda_neigh_bar")

# Row 3: Correlation & Property Types
col1, col2 = st.columns(2)

with col1:
    st.header("Feature Correlation")
    # Using raw features available in the original CSV
    corr_cols = ["price", "accommodates", "bathrooms", "bedrooms", "number_of_reviews"]
    corr_matrix = df[corr_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True, key="eda_corr_heat")

with col2:
    st.header("Property Hierarchy")
    fig_sun = px.sunburst(
        df,
        path=['property_type_grouped', 'property_type'],
        values='price',
        color='property_type_grouped'
    )
    st.plotly_chart(fig_sun, use_container_width=True, key="eda_prop_sun")

# --- 5. INSIGHTS ---
st.divider()
st.header("Key Insights")
st.markdown("""
- **City Trends:** SF and DC are the most expensive markets.
- **Capacity:** 'Accommodates' and 'Bedrooms' show the strongest positive correlation with price.
- **Hierarchy:** Most listings are grouped under 'Apartment' or 'House' categories.
""")