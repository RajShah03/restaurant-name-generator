import streamlit as st
import langchain_helper

st.title("🍴 Restaurant Name & Menu Generator")

# Sidebar for cuisine selection
cuisine = st.sidebar.selectbox(
    "Pick a cuisine",
    ("Indian", "Italian", "Mexican", "Arabic", "American")
)

if cuisine:
    response = langchain_helper.generate_restaurant_names_and_items(cuisine)

    # Show restaurant name
    st.header(response['restaurant_name'])

    # Show menu items
    st.subheader("Menu Items")
    for item in response['menu_items']:  # already a list
        st.write("-", item)
