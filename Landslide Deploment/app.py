import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

pickle_in= open('LandslideRF.pkl', 'rb')
model = pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

def predict_landslide(input):
    prediction = model.predict(input)
    return prediction

def main():
    # Define the layout of the app
    st.title('Landslide Prediction in Bhutan')
    st.write('Details to check the possibility of landslide:')

    # Input fields for the user to enter data
    lithology = st.number_input('Lithology')
    altitude = st.number_input('Altitude')
    aspect = st.number_input('Aspect')
    distance_to_road = st.number_input('Distance to road')
    distance_to_stream = st.number_input('Distance to stream')


    # Dictionary to assign input data
    input_data = {'Lithology': lithology, 'Altitude': altitude, 'Aspect': aspect, 'Distance to road': distance_to_road, 'Distance to stream': distance_to_stream}
    
    # Dictionary to Df
    input_df = pd.DataFrame(input_data, index=[0])
    
    # train data columns
    input_df = input_df[['Lithology', 'Altitude', 'Aspect', 'Distance to road', 'Distance to stream']]
    
    if st.button('Predict'):
        prediction = predict_landslide(input_df)
        if prediction == 1:
            st.write('Slide')
        else:
            st.write('Non slide')

if __name__ == '__main__':
    main()