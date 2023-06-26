import streamlit as st
from streamlit_ace import st_ace

first, second = st.beta_columns(2) 
# Display editor's content as you type

with first:
    # Spawn a new Ace editor
    content = st_ace()

with second:
    st.write(content)