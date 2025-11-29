import streamlit as st
import requests
import json

API_URL = 'http://127.0.0.1:8000/predict'
INGREDIENTS_PATH = '../data/ingredients.json'

st.set_page_config(page_title='CUISINE RECOMMENDER', layout='centered')

@st.cache_data(show_spinner=False)
def load_ingredients():
    with open(INGREDIENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

st.title(body='CUISINE RECOMMENDER')

all_ingredients = load_ingredients()

if all_ingredients:
    selected_ingredients = st.multiselect(
        label='Choose your ingredients',
        options=all_ingredients,
        placeholder='Find ingredients',
    )

    if not selected_ingredients:
        st.warning(body='Please select ingredients to receive a cuisine recommendation')

    if selected_ingredients:
        predict_button = st.button(
            label='Suggest cuisine',
            width='stretch',
        )

        if predict_button:
            try:
                with st.spinner(text='In progress'):
                    payload = {"selected_ingredients": selected_ingredients}
                    response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()

                    cuisine = result['cuisine']
                    confidence = result['confidence']

                    st.success(body='Enjoy the meal!')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label='Cuisine suggestion', value=cuisine)
                    with col2:
                        st.metric(label='Confidence score', value=f'{confidence * 100:.2f}%')

                else:
                    st.error(body=f'Server error ({response.status_code}): {response.text}')

            except requests.exceptions.ConnectionError:
                st.error(body='Unable to connect to the API server')
