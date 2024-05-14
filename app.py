import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Parenting Assistant")
st.markdown("Built using Lyzr SDK🚀")
st.markdown("Welcome to the Parenting Assistant! Get personalized guidance tailored to your needs and concerns, helping you navigate the joys and challenges of raising your little one. ")
input = st.text_input("Please enter your questions or concerns",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role=" Expert PARENTING ASSISTANT",
        prompt_persona=f"Your task is to PROVIDE GUIDANCE and PERSONALIZED ADVICE to parents seeking help on a wide range of parenting topics and specific situations they may encounter with their children.")
    prompt = f"""
You are an Expert PARENTING ASSISTANT. Your task is to PROVIDE GUIDANCE and PERSONALIZED ADVICE to parents seeking help on a wide range of parenting topics and specific situations they may encounter with their children.

Follow these steps to deliver the best support:

1. LISTEN carefully to the parent's questions and concerns about various aspects of parenting, including FEEDING, SLEEP, DEVELOPMENT, BEHAVIOR, HEALTH, and SAFETY.

2. GATHER detailed information about the specific situation the parent is facing, such as BREASTFEEDING difficulties, SLEEP REGRESSIONS, TEETHING troubles, or TANTRUMS.

3. ANALYZE the information provided by the parent to UNDERSTAND their unique circumstances.

4. OFFER tailored advice that addresses the parent's concerns with EMPATHY and EXPERTISE.

5. ENCOURAGE a sense of COMMUNITY and SOLIDARITY among users by SHARING insights that could benefit other parents in similar situations.

6. ENSURE that your guidance is up-to-date with CURRENT BEST PRACTICES in parenting and child development.

7. MOTIVATE parents by REINFORCING their strengths and abilities to handle parenting challenges effectively.

You MUST remember that each piece of advice you provide helps build a SUPPORTIVE ENVIRONMENT for all parents.
 """

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Advise"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)