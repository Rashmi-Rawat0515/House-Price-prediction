import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

#page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

#Load and train model
@st.cache_resource
def train_model():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target

    x = df.drop("Price", axis=1)
    y = df["Price"]

    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, r2, df, rmse, scaler

model, r2, df, rmse, scaler = train_model()

#title
st.title("Calfornia House Price Predictor")
st.markdown("Enter house details to get an instant price prediction using Machine Learning")

st.divider()

#prediction form
st.subheader("Enter House Details")
col1, col2, col3 = st.columns(3)
with col1:
    med_inc = st.slider("Meadian Income (area)",0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("House Age(years)",1, 52, 20)
    ave_rooms = st.slider("Average Rooms",1.0, 10.0, 5.0, 0.1)

with col2:
    ave_bedrms = st.slider("Average Bedrooms", 1.0, 5.0, 2.0, 0.1)
    population = st.slider("Area Population", 100, 10000, 1500)
    ave_occup = st.slider("Average Occupancy", 1.0, 6.0, 3.0, 0.1)

with col3:
    latitude = st.slider("Latitude", 32.5, 42.0, 35.0, 0.1)
    longitude = st.slider("Longitude", -124.0, -114.0, -119.0, 0.1)

#predict
input_data = np.array([float(med_inc), float(house_age), float(ave_rooms), 
                        float(ave_bedrms), float(population), float(ave_occup), 
                        float(latitude), float(longitude)]).reshape(1, -1)
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

st.divider()

# show prediction
st.subheader("Predicted Price")
col_a,col_b,col_c = st.columns(3)
col_a.metric("Estimated Price",f"${prediction*100000:.0f}")
col_b.metric("Model R2 Score",f"{round(r2, 3)}")
col_c.metric("Model RMSE Score",f"{round(rmse, 3)}")

st.divider()

#Feature importance
st.subheader("What Drives Your House Prices?")

feature_importance = pd.Series(model.feature_importances_,
                               index=df.drop("Price",axis=1).columns)
feature_importance = feature_importance.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="Oranges_r", ax=ax)
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance")
st.pyplot(fig)

st.divider()

#Raw Data
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df.head(50))
