import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
from sklearn.preprocessing import OrdinalEncoder

@st.cache_resource
# Load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)
    
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: contain;
            background-position: fit;
            background-repeat: repeat;
            background-attachment: fixed;
        }}     
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local(r"12.png")

model=load_model("model_car.pkl")

encoder_city=load_model("encoder_city.pkl")
encoder_Insurance_Validity=load_model("encoder_Insurance_Validity.pkl")
encoder_bt=load_model("encoder_bt.pkl")
encoder_ft=load_model("encoder_ft.pkl")
encoder_oem=load_model("encoder_oem.pkl")
encoder_model=load_model("encoder_model.pkl")
encoder_transmission=load_model("encoder_transmission.pkl")
encoder_variantName=load_model("encoder_variantName.pkl")

ml_df=pd.read_excel("ml_dl.xlsx")
st.title("Car Price Prediction App")

categorical_features = ["city", "ft", "bt", "transmission", "oem", "model", "variantName", "Insurance Validity"]
dropdown_options = {feature: ml_df[feature].unique().tolist() for feature in categorical_features}

tab1, tab2 = st.tabs(["Home", "Predict"])
with tab1:
    st.markdown("""
                **1. Introduction**
                In the rapidly evolving automotive market, determining the right price for a vehicle is crucial 
                for both buyers and sellers. The Car Price Prediction App provides an intelligent solution to 
                estimate car prices based on key parameters using machine learning models. This tool helps users 
                make data-driven decisions by leveraging historical data and predictive analytics.
                
                **2. Problem Statement**
                Buying or selling a car requires understanding its fair market value, which is influenced 
                by multiple factors such as brand, model, year of manufacture, mileage, fuel type, and transmission. 
                Manually evaluating these factors can be complex and time-consuming. The Car Price Prediction App simplifies 
                this process by providing instant and accurate price predictions.
                
                **3. Key Features**
                User-Friendly Interface: Simple and interactive Streamlit-based UI.
                Machine Learning Model: Utilizes an advanced regression model ( Random Forest) trained 
                on a vast dataset of car prices.
                Feature Inputs: Users can enter details like car brand, model, manufacturing year, fuel type, 
                transmission, and other relevant attributes.
                Real-Time Predictions: Provides instant car price estimates based on input parameters.
                Data Visualization: Includes graphs and charts to show trends in car pricing.
                Comparison Tool: Allows users to compare multiple cars for better decision-making.
                
                **4. Target Audience**
                Car Buyers & Sellers: Individuals looking to buy or sell a used car at a fair market price.
                Dealerships & Businesses: Car dealerships and resellers who need an efficient way to estimate car values.
                Financial Institutions: Banks and insurance companies that assess car values for loan and policy decisions.
                
                **5. Technologies Used**
                Frontend: Streamlit for an interactive and user-friendly web application.
                Backend: Python with machine learning libraries such as Scikit-learn, XGBoost, and Pandas.
                Model Deployment: Trained ML model integrated into the Streamlit app for real-time predictions.
                Visualization: Matplotlib and Seaborn for trend analysis and price distribution.
                
                **6. Benefits**
                Saves time by providing instant price predictions.
                Eliminates guesswork in car pricing.
                Empowers users with data-driven insights for negotiation.
                Enhances transparency in the used car market.
                
                **7. Conclusion**
                The Car Price Prediction App is a powerful tool for individuals and businesses looking to evaluate 
                car prices efficiently. By leveraging machine learning, it offers a seamless experience in determining a 
                car's fair value, making the buying and selling process more transparent and informed.
                """)
with tab2:
    a1,a2,a3=st.columns(3)
    a4,a5,a6=st.columns(3)
    a7,a8,a9=st.columns(3)
    a10,a11,a12=st.columns(3)
    a13,a14=st.columns(2)
    
    with a1:
        city_select=st.selectbox("Select City",dropdown_options["city"])
        city=encoder_city.transform([[city_select]])[0][0]
    with a2:
        ft_select=st.selectbox("Select fuel Type",dropdown_options["ft"])
        ft=encoder_ft.transform([[ft_select]])[0][0]
    with a3:
        bt_select=st.selectbox("Select Body Type",dropdown_options["bt"])
        bt=encoder_bt.transform([[bt_select]])[0][0]
    with a4:
        km=st.number_input("Enter KM driven",min_value=10)
    with a5:
        transmission_select=st.selectbox("Select Body Type",dropdown_options["transmission"])
        transmission=encoder_transmission.transform([[transmission_select]])[0][0]
    with a6:
        ownerNo=st.number_input("Enter no. of Owner's",min_value=1)
    with a7:
        oem_select=st.selectbox("Select car manufacture name",dropdown_options["oem"])
        oem=encoder_oem.transform([[oem_select]])[0][0]
    with a8: 
        model_select=st.selectbox("Select car Model name",dropdown_options["model"])
        model=encoder_model.transform([[model_select]])[0][0]
    with a9:
        modelYear=st.number_input("Enter car manufacture year",min_value=1000)
    with a10:
        variantName_select=st.selectbox("Select Model variant Name",dropdown_options["variantName"])
        variantName=encoder_variantName.transform([[variantName_select]])[0][0]
    with a11:
        Registration_Year=st.number_input("Enter car registration year",min_value=1000)
    with a12:
        InsuranceValidity_select=st.selectbox("Select Insurance Type",dropdown_options["Insurance Validity"])
        InsuranceValidity=encoder_Insurance_Validity.transform([[InsuranceValidity_select]])[0][0]
    with a13:
        Seats=st.number_input("Enter seat capacity",min_value=1)
    with a14:
        EngineDisplacement=st.number_input("Enter Engine CC",min_value=1)
        
    if st.button('Predict'):
        input_data = pd.DataFrame([{
            "city": city,
            "ft": ft,
            "bt": bt,
            "km": km,
            "transmission": transmission,
            "ownerNo": ownerNo,
            "oem": oem,
            "model": model,
            "modelYear": modelYear,
            "variantName": variantName,
            "Registration Year": Registration_Year,
            "Insurance Validity": InsuranceValidity,
            "Seats": Seats,
            "Engine Displacement": EngineDisplacement
        }])

        a=["km","ownerNo","modelYear","price","Registration Year","Seats","Engine Displacement"]
        input_data[a]=np.cos(input_data[a])
        prediction = model.predict(input_data)
        
        st.subheader("Predicted Car Price")
        st.write(f"₹ {prediction[0]:,.2f}")
