from predictionguard import PredictionGuard
import streamlit as st

client = PredictionGuard()


#---------------------#
# Streamlit config    #
#---------------------#

#st.set_page_config(layout="wide")

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#----------------------------#
#   Streaming setup          #
#----------------------------#

def stream_tokens(model, messages):
    for sse in client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        stream=True
    ):
        yield sse["data"]["choices"][0]["delta"]["content"]


#--------------------------#
# Streamlit sidebar        #
#--------------------------#

st.sidebar.markdown(
    "This chat interface uses [Prediction Guard](https://www.predictionguard.com) LLMs to carry on a convo with users."
)


#--------------------------#
# Streamlit app            #
#--------------------------#

system_message = "You are a helpful assistant. You only answer in spanish. Do not answer in English no matter what. Only Spanish. Please only spanish."

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the system prompt at the start of the session if not already present
if not st.session_state.messages or (st.session_state.messages and st.session_state.messages[0].get("role") != "system"):
    st.session_state.messages.insert(0, {"role": "system", "content": system_message})

for message in [msg for msg in st.session_state.messages if msg["role"] != "system"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Message the bot..."):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        response = st.write_stream(stream_tokens(
                "Hermes-3-Llama-3.1-8B", 
                st.session_state.messages
            ))
        
    st.session_state.messages.append({"role": "assistant", "content": response})