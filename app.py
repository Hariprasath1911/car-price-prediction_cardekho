import streamlit as st
import pickle
import pandas as pd
import base64

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
            background-size: cover;
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
encoder=load_model("ordinal_encoder.pkl")

ml_df=pd.read_excel("extracted_car_details.xlsx")


st.title("Car Price Prediction App")

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
    a13,a14=st.columns(3)
    
    with a1:
        car_city = ml_df["city"].unique().tolist()
        city_select=st.selectbox("Select City",car_city)
        city = encoder.transform([[city_select]])[0][0]
    with a2:
        car_ft = ml_df["ft"].unique().tolist()
        ft_select=st.selectbox("Select fuel Type",car_ft)
        ft=encoder.transform([[ft_select]])[0][0]
    with a3:
        car_bt=ml_dl["bt"].unique().tolist()
        bt_select=st.selectbox("Select Body Type",car_bt)
        bt=encoder.transform([[bt_select]])[0][0]
    with a4:
        km=st.number_input("Enter KM driven",min_value=10)
    with a5:
        car_transmission=ml_dl["transmission"].unique().tolist()
        transmission_select=st.selectbox("Select Body Type",car_transmission)
        transmission=encoder.transform([[transmission_select]])[0][0]
    with a6:
        ownerNo=st.number_input("Enter no. of Owner's",min_value=1)
    with a7:
        car_oem=ml_dl["oem"].unique().tolist()
        oem_select=st.selectbox("Select car manufacture name",car_oem)
        oem=encoder.transform([[oem_select]])[0][0]
    with a8:
        car_model=ml_dl["model"].unique().tolist()
        model_select=st.selectbox("Select car Model name",car_model)
        model=encoder.transform([[model_select]])[0][0]
    with a9:
        modelYear=st.number_input("Enter car manufacture year",min_value=1000)
    with a10:
        car_variantName=ml_dl["variantName"].unique().tolist()
        variantName_select=st.selectbox("Select Model variant Name",car_variantName)
        variantName=encoder.transform([[variantName_select]])[0][0]
    with a11:
        Registration_Year=st.number_input("Enter car registration year",min_value=1000)
    with a12:
        car_InsuranceValidity=ml_dl["Insurance Validity"].unique().tolist()
        InsuranceValidity_select=st.selectbox("Select Insurance Type",car_InsuranceValidity)
        InsuranceValidity=encoder.transform([[InsuranceValidity_select]])[0][0]
    with a13:
        Seats=st.number_input("Enter seat capacity",min_value=1)
    with a14:
        EngineDisplacement=st.number_input("Enter Engine CC",min_value=1)
        
if st.button('Predict'):
    data={
        "city":city,"ft":ft,"bt":bt,"km":km,"transmission":transmission,"ownerNo":ownerNo,"oem":oem,"model":model,"modelYear":modelYear,
        "variantName":variantName,"Registration Year":Registration_Year,"Insurance Validity":InsuranceValidity,"Seats":Seats,
        "Engine Displacement":EngineDisplacement
        }
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    st.subheader("Predicted car Price")
    st.write(f"₹ {prediction[0]:,.2f}")
