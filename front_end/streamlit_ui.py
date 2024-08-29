import streamlit as st
import time, random, json, asyncio, requests
import regex as re

def generate_reply(prompt):
    try:
        response = requests.post(url="http://127.0.0.1:8000/prompt", data=json.dumps({"input": prompt}))
    except Exception as e:
        response = {
            'content': None,
            'error': str(e)
        }

    return response

st.title("Your personal AI dietitian!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! How can i help you?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Make your diet plan today!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(f"{prompt}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        resp = generate_reply(prompt)

        if type(resp) == dict and resp['error']:
            st.write("An error has occurred please let us know if the issue persists")

        else:
            resp = resp.json()
            st.write(resp)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": resp})



