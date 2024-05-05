from dotenv import load_dotenv
import streamlit as st
import seaborn as sns
from langchain_anthropic import ChatAnthropic
from pandasai import SmartDataframe, Agent  
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from pandasai.llm.local_llm import LocalLLM
from pandasai.llm.google_gemini import GoogleGemini


load_dotenv()

# Set environment variable for PandasAI API key
os.environ["PANDASAI_API_KEY"] = os.getenv('pandasai_api')

#llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose = True, temperature = 0.5, google_api_key=os.getenv('gemini_api'))
llm = GoogleGemini(api_key=os.getenv('gemini_api'), model="gemini-pro")

st.title("Data visualization with PandasAI")

uploaded_file = st.file_uploader("Upload a csv file " , type = ['csv'])
if uploaded_file is not None :
    #data = sns.load_dataset("penguins")
    data = pd.read_csv(uploaded_file)

    st.write(data.head(3))


    agent = Agent(data)
    #df =SmartDataframe(data , config={"llm":agent})
    prompt = st.text_area("Enter your prompt:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):

                st.write(agent.chat(prompt ))  
