import streamlit as st
import pickle
import numpy as np

# Load the pickle files
num_impute_model = pickle.load(open('num_impute_pickle.pkl', 'rb'))
cat_impute_model = pickle.load(open('cat_impute_pickle.pkl', 'rb'))
standardization_model = pickle.load(open('std_pickle.pkl', 'rb'))
encoding_model = pickle.load(open('encode_pickle.pkl', 'rb'))
pca_model = pickle.load(open('pca_pickle.pkl', 'rb'))
model = pickle.load(open('rfc_model_pickle.pkl', 'rb'))

# Function to preprocess the input attributes
def predict(item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_size, outlet_location_type, outlet_type):
    # Imputation
    num_var=[[item_weight,item_visibility,item_mrp]]
    num_impute = num_impute_model.transform(num_var)

    cat_var=[[item_fat_content,item_type,outlet_size,outlet_location_type,outlet_type]]
    cat_impute = cat_impute_model.transform(cat_var)

    from scipy.sparse import csr_matrix

       # Standardization
    std_var = standardization_model.transform(num_impute)

    # Encoding
    encode_var = encoding_model.transform(cat_impute)
    if isinstance(encode_var, csr_matrix):
        encode_var = encode_var.toarray()

    input_data = np.concatenate((std_var, encode_var), axis=1)




    # PCA
    pca_var = pca_model.transform(input_data)

    prediction = model.predict(pca_var)
    return prediction


# Streamlit app
def main():
    import streamlit_theme as stt
    # Read the contents of the CSS file
    with open('style.css') as f:
        css = f.read()

    # Add the CSS to the app's page
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.title("BIGG MART ITEM SALES PREDICTION") 

    # Create input fields for each attribute
    item_weight = st.number_input("Item Weight")
    item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
    item_visibility = st.number_input("Item Visibility")
    item_type = st.selectbox("Item Type", ["Fruits and Vegetables", "Snack Foods", "Household","Frozen Foods","Dairy","Canned","Baking Goods","Health and Hygiene","Soft Drinks","Meat","Breads","Hard Drinks","Others","Starchy Foods","Breakfast","Seafood"])
    item_mrp = st.number_input("Item MRP")
    outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3","Grocery Store"])

    # Make predictions on button click
    if st.button("predict"):
        # Make predictions
        predictions = predict(item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_size, outlet_location_type, outlet_type)
        st.subheader("Hello,...! We are Very Happy for Using Our Service.")

        st.write("Based on the Provided Details, We are Predicting that the Sales For this Product is :")
        st.subheader(predictions[0])
  

if __name__ == "__main__":
    main()
