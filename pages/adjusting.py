import streamlit as st
import numpy as np
from os.path import dirname, join 
import sys 
sys.path.insert(0, join(dirname(__file__), '..')) 

from ratemaking import net_prem_earned

st.set_page_config(page_title="Adjusting", page_icon="🎢")

net_prem_earned