"""
streamlit run app.py
"""

import os
import requests
import tempfile
import streamlit as st
from streamlit_chat import message
from datetime import datetime, timedelta

from utils.m0_query_llama_planning import s1_generate_budget, s2_suggest_activities
from utils.m1_chat_persona import AI_chat_responder, llm_conversation_model
from m3_chat_context import llm_context_setup, llm_context_query

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        body {
            background-color: #cca7fa;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.session_state.use_my_data_flag = False

default_start_date = datetime.today() + timedelta(days=15)
default_end_date = datetime.today() + timedelta(days=25)
default_response = "Please submit your travel details, and hit **Plan my vacation**!"


def m4_live_chatbot():
    """
    Live chatbot engine
    """
    # Output Section
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = st.text_input("💬 Ask me anything about your trip!", key='input')

    if user_input:
        if st.session_state["use_my_data_flag"] and "vectorstore_faiss_aws" in st.session_state:
            output = llm_context_query(user_input, st.session_state['vectorstore_faiss_aws'])
        else:
            # output = "Something random!"
            output = AI_chat_responder(user_input, st.session_state["conversation_chain"])
        # store the output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(str(output))
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

def main():
    """
    Application main module
    """
    # Input Section
    st.title("Travel Helpdesk")
    st.subheader("Please enter your travel details!")

    # Creating a 2x3 matrix for input fields
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        origin_country = st.text_input("From Country", "India")
        travel_dates_start = st.date_input("From", default_start_date)
        num_adults = st.number_input("Number of Adults", min_value=1, value=1)
    with col2:
        destination_country = st.text_input("To Country", "United Arab Emirates")
        travel_dates_end = st.date_input("To", default_end_date)
        num_kids = st.number_input("Number of Kids", min_value=0, value=0)
    with col3:
        destination_cities = st.text_input("Cities to explore (optional)", "best cities")
        # Add LLM selection UI
        st.markdown("#### 🤖 Select the LLM")
        llm_options = [
            'amazon.titan-tg1-large',
            'amazon.titan-text-express-v1',
            'amazon.titan-text-lite-v1',
            'anthropic.claude-v2',
            'anthropic.claude-instant-v1',
        ]

        selected_llm = st.radio("Choose any one",
                                options=llm_options)
        if "conversation_chain" not in st.session_state:
            st.session_state["conversation_chain"] = llm_conversation_model(selected_llm)
        st.session_state["conversation_chain"] = llm_conversation_model(selected_llm)
    with col4:
        st.subheader("🗣️ Live Chat")
        use_my_data_flag = st.checkbox('Tick if you have your own data file!', False)

        st.session_state['use_my_data_flag'] = use_my_data_flag
        if st.session_state["use_my_data_flag"]:
            csv_docs = st.file_uploader(
                "Upload your CSV data file and Process", type="csv", accept_multiple_files=False)
            if st.button("Process"):
                with st.spinner("Processing"):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(csv_docs.getvalue())
                        tmp_file_path = tmp_file.name
                        if "vectorstore_faiss_aws" not in st.session_state:
                            st.session_state["vectorstore_faiss_aws"] = llm_context_setup(tmp_file_path)
                        print(f"vectorstore_faiss_aws:created={st.session_state.vectorstore_faiss_aws}::")

        # Initialize chatbot
        m4_live_chatbot()

    submit = st.button("Plan my vacation!")

    ## Itinerary Output
    num_days = (default_end_date - default_start_date).days

    plan_budget = default_response
    plan_activity = default_response

    if submit==True :
        # Model 1 : Llama
        plan_budget = s1_generate_budget(origin_country, travel_dates_start, travel_dates_end, destination_country,
                                                 destination_cities, num_adults, num_kids)
        plan_activity = s2_suggest_activities(destination_country, travel_dates_start, travel_dates_end,
                                         destination_cities, num_adults, num_kids)

        # Alternate Model 2 : Titan
        # plan_budget = AI_chat_responder(plan_budget, st.session_state["conversation_chain"])
        # plan_activity = AI_chat_responder(plan_activity, st.session_state["conversation_chain"])

    # Start Planning!
    if submit:
        st.header("Here is your Itinerary!")
        st.subheader(
            "Booking for {destination_country} : {D} Days {N} Nights".format(destination_country=destination_country,
                                                                             D=num_days, N=num_days - 1))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("💸 Estimated Budget")
            st.info(plan_budget)
        with col2:
            st.subheader("✅ Suggested Activities")
            st.success(plan_activity)
        # with col3:
        #     st.subheader("🗣️ Live Talk")

    # Display relevant image
    # st.header("Relevant Image:")
    # display_relevant_image()


if __name__ == "__main__":
    main()