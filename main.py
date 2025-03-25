from crewai import Agent, Task, Crew,LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
import streamlit as st
import streamlit_ext as ste

import os
from IPython.display import Markdown
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit


def generate_pdf(text):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    y = 750  # Start position for text
    max_width = 500  # Maximum width for wrapping text

    for line in text.split("\n"):
        if line.startswith("# "):  # Markdown header (H1)
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(50, y, line[2:])
        elif line.startswith("## "):  # Markdown subheader (H2)
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(50, y, line[3:])
        elif line.startswith("- "):  # Bullet points
            pdf.setFont("Helvetica", 12)
            wrapped_text = simpleSplit("• " + line[2:], "Helvetica", 12, max_width)
            for sub_line in wrapped_text:
                pdf.drawString(50, y, sub_line)
                y -= 20
            continue
        else:  # Normal text
            pdf.setFont("Helvetica", 12)
            wrapped_text = simpleSplit(line, "Helvetica", 12, max_width)
            for sub_line in wrapped_text:
                pdf.drawString(50, y, sub_line)
                y -= 20

        y -= 20  # Move to the next line

        if y < 50:  # New page if needed
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = 750

    pdf.save()
    buffer.seek(0)
    return buffer

st.set_page_config(page_title="👨‍🏫 AI Teaching Agent", layout="centered")

with st.sidebar:
    st.session_state['gemini_key'] = st.text_input("Please Enter your Gemini key")
    st.session_state['SERPER_API_KEY'] = st.text_input("Please Enter your SERPER API key")


if not st.session_state['gemini_key'] or not st.session_state['SERPER_API_KEY']:
    st.error("Please enter your Gemini and SERPER API keys in the sidebar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.session_state['gemini_key']
os.environ["SERPER_API_KEY"] = st.session_state['SERPER_API_KEY']

llm = LLM(
    model='gemini/gemini-1.5-flash',
    api_key=st.session_state['gemini_key']

)


llm2 = LLM(
    model='gemini/gemini-2.0-flash-exp',
    api_key=st.session_state['gemini_key']
)



planner = Agent(
    role="Study Planner",
    goal="To build a comprehensive knowledge base and detailed learning roadmap for a given topic, catering to the user's specified understanding level {level}",
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    backstory="The user seeks to master complex concepts about {topic}"
    "They require an in-depth guide that covers fundamental concepts, advanced techniques, and recent developments."
    "The solution must begin with first principles, ensuring clarity for beginners while still offering depth for experienced learners."
    "The roadmap should enable learners to progress from foundational knowledge to expert-level mastery in a structured and logical way."
    "The user has access to SerperDevTool for web search and ScrapeWebsiteTool for content extraction, ensuring access to updated information."
    "Your work is the basis for "
    "the Content Writer to Create study material.",
    llm=llm2,
    allow_delegation=False,
	verbose=True
)


writer = Agent(
    role="Content Writer",
    goal="The goal is to create a comprehensive, well-structured study material for a given topic: {topic} that aligns with the provided content outline. The final content should include a curated list of high-quality learning resources, with clear descriptions and reference links for each.",
    tools=[SerperDevTool(), WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini/gemini-1.5-flash",
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)],
    backstory="The user aims to create a structured and engaging study guide for learners seeking in-depth knowledge about a particular {topic}."
              "The Content Planner outlines the key objectives and learning path, ensuring the material aligns with educational goals." 
            "The user has access to the SerperDevTool and WebsiteSearchTool, which the agent can leverage to fetch updated and credible learning resources."
              ,
    llm = llm,
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Create the Knowledge Base: Start from first principles and explain core concepts about {topic} with clear definitions and examples. Introduce key terminology, core principles, and relevant frameworks. Cover practical applications, best practices, and emerging trends.\n"
        "2. Identify user's familarity level {level}, and generate plan according to that \n"
        "3. Break the topic into logical subtopics. Structure the subtopics in a progressive order, starting with fundamentals and advancing to expert-level concepts. Recommend resources such as books, articles, video tutorials, and interactive exercises. \n"
        "4. Gather updated information from trusted sources to ensure content reflects the latest trends, research, and advancements about {topic}"

    ),
    expected_output="A comprehensive stydy plan"
        "Detailed explanations of core concepts, key terminology, frameworks, and best practices."
        "A structured progression of topics, divided into foundational, intermediate, and advanced levels.",
    agent=planner,
)

write = Task(
    description=(
        "1. Write new study material about {topic} that aligns with the provided content outline and context shared by the Content Planner.Maintain clarity, coherence, and educational value in the content.\n"
		"2. Use the SerperDevTool to search for technical blogs, GitHub repositories, official documentation, video tutorials, and courses.Use the WebsiteSearchTool to fetch additional in-depth resources if needed. \n"
        "3. Organize resources in a well-structured format. Each entry should include:"
       "Title/Name of the resource."
        "A brief description explaining its relevance and key takeaways."
        "A direct link for easy access.\n"
        "4. Present the content in a clean, professional, and organized structure. Use headings, bullet points, and clear descriptions to improve readability.\n"

    ),
    expected_output="A well-written study material"
        "A detailed study material aligned with the content outline."
        "A well-curated list of resources for deeper learning."
        "Each resource should include a clear description and reference link.Content should be presented neatly with clear formatting (e.g., headings, bullet points, and structured paragraphs). DO NOT MENTION TOOL NAME IN YOUR RESPONSE"
        "IMPORTANT, MAKE SURE OUTPUT SHOULD BE IN PROPER MARKDOWN FORMAT",
    agent=writer,
)




crew = Crew(
    agents=[planner, writer],
    tasks=[plan, write],
    verbose=True
)



teaching_assistant_agent2  = Agent(
    role="Exercise Creator2",
    goal="The goal is to create comprehensive and engaging practice materials for a given topic, ensuring they align with a structured roadmap :{plan}. The materials should include a variety of exercises, quizzes, hands-on projects, and real-world application scenarios. Detailed solutions and explanations must be provided for all practice materials to facilitate learning and self-assessment.",
    tools=[SerperDevTool(),ScrapeWebsiteTool()],
    backstory="As part of an educational initiative, there is a need to develop high-quality practice materials for learners studying a specific topic : {topic}. These materials should not only reinforce theoretical knowledge but also provide practical, real-world applications to enhance understanding and retention. The materials must be progressive, starting from foundational concepts and advancing to more complex problems and projects. To ensure relevance and accuracy, the materials will be sourced and validated using tools like SerpApi",
    llm=llm,
    allow_delegation=False,
	verbose=True
)

quiz2 = Task(
    description=(
        "1. Progressive Exercises: A series of exercises that start with basic concepts and gradually increase in complexity.\n"
        "2. Hands-on Projects: Practical projects that require application of the topic in real-world scenarios.\n"
        "3. Real-world Application Scenarios: Case studies or examples that demonstrate how the topic is applied in real-world situations.\n"
        "4. The materials should be aligned with a structured roadmap progression, ensuring that learners can follow a logical path from basic to advanced concepts. The use of SerpApi tool will help in finding example problems, real-world applications, and relevant content to include in the practice materials."

    ),
    expected_output="Progressive Exercises: A document or set of documents containing a series of exercises, starting from basic to advanced levels."
        " Hands-on Projects: Detailed project descriptions and guidelines for hands-on projects, including expected outcomes and evaluation criteria."
        "Real-world Application Scenarios: Case studies or examples demonstrating real-world applications of the topic, with explanations."
        "The final output should be well-organized, easy to follow, and tailored to the given topic, ensuring that learners can effectively practice and apply their knowledge."
        "DO NOT WoRRY ABOUT TIME TAKEN and LENGHT Constraint, PROVIDE EXErcises for complete roadmap and DO NOT MENTION tool names",

    agent=teaching_assistant_agent2,
)


crew3 = Crew(
    agents=[teaching_assistant_agent2],
    tasks=[quiz2],
    verbose=True
)


st.title("👨‍🏫 AI Teaching Agent Team")
st.markdown("Enter a topic to generate a detailed learning path and resources")

# Add info message about Google Docs
st.info("📝 The agents will create detailed learning roadmap")

# Query bar for topic input

st.session_state['topic'] = st.text_input("Enter the topic you want to learn about:", placeholder="e.g., Machine Learning, LoRA, etc.")
st.session_state['level'] = st.selectbox("what is  your familarity level:", ('Beginner', 'Intermediate','Advance'))

results = []

if st.button("Start"):
    if not st.session_state['gemini_key']:
        st.error("Please enter a topic.")
    
    elif not st.session_state['topic']:
        st.error("Please enter a topic.")
    elif not st.session_state['topic']:
        st.error("Please enter level.")
    else:
        with st.spinner("Generating RoadMap"):
            result = crew.kickoff(inputs={"topic": st.session_state['topic'], "level":st.session_state['level']})
            results.append(result)
        st.markdown("### Roadmap")
        st.markdown(result.raw)
        st.divider()
        pdf_buffer = generate_pdf(result.raw)
        ste.download_button(label="Export_RoadMap",
                    data=pdf_buffer,
                    file_name="NLP_Study_Guide.pdf",
                    mime='application/pdf')
        
        st.divider()

   
        with st.spinner("Generating Assesment to Practice"):
            result3 = crew3.kickoff(inputs={"topic": st.session_state['topic'], "plan":result.raw})
            st.markdown("### Assesment")
            st.markdown(result3.raw)
            st.divider()
            pdf_buffer = generate_pdf(result3.raw)
            st.download_button(label="Export_Assesment",
                    data=pdf_buffer,
                    file_name="Assesment.pdf",
                    mime='application/pdf')


            







