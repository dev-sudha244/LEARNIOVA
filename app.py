import os
import requests
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import urllib.parse
from io import BytesIO
from typing import List, Optional, Union, Dict, Any
import base64
import json 

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 


# --- Configuration for Pre-set Resources (Categorized) ---
PRESET_RESOURCES = {
    "PYQ's 📝": {
        "PYQ's - 24-ECOnomics Exam Paper": "https://drive.google.com/uc?export=download&id=1O_PtykOXqmX3ut8JhL3y0YZtAhlY8Vj0",
        "PYQ's - 2023 Exam Paper": "...",
    },
    "Teacher Notes 🧑‍🏫": {
        "Teacher Notes - Basic electrical": "......", 
        "Teacher Notes - Advanced OOP": "....",
    },
    "Senior Notes 🎓": {
        "Senior Notes - RAG Deep Dive": "....",
        "Senior Notes - System Design": ".....",
    }
    
}
# -----------------------------

# -----------------------------
# 🔐 Configuration and Helpers
# -----------------------------
def configure_genai():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"⚠️ Google API key missing or invalid: {e}")
        return False
    
def download_from_drive(url):
    """Downloads a file from a Google Drive direct link and returns a BytesIO buffer."""
    try:
        if not ("drive.google.com" in url and "id=" in url): return None
        file_id = url.split("id=")[-1].split("&")[0]
        response = requests.get(f"https://drive.google.com/uc?export=download&id={file_id}")
        if response.status_code == 200 or response.status_code == 302: return BytesIO(response.content)
        return None
    except Exception: return None

def extract_text_from_pdf(pdf_file_or_buffer):
    """Extracts text content from a single PDF file buffer."""
    try:
        reader = PdfReader(pdf_file_or_buffer)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text.strip() if text.strip() else None
    except Exception as e:
        return None

@st.cache_data(show_spinner="Generating Mindmap...")
def create_mindmap_markdown(text):
    """Generates a hierarchical markdown mindmap from text using Gemini."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"Create a markdown mindmap of this content using headings (#, ##, ###) and bullet points (-). Respond ONLY in markdown. Text: {text[:30000]}"
    try:
        return model.generate_content(prompt).text.strip()
    except Exception: return None

def create_markmap_html(markdown_content):
    """Creates the Markmap HTML embed code for interactive visualization."""
    markdown_content = markdown_content.replace("`", "\\`").replace("${", "\\${")
    html = f"""
    <!DOCTYPE html><html><head><meta charset="UTF-8"><style>body {{ margin: 0; overflow: hidden; background: #fff; }} #mindmap {{ width: 100%; height: 100vh; }}</style><script src="https://cdn.jsdelivr.net/npm/d3@6"></script><script src="https://cdn.jsdelivr.net/npm/markmap-view"></script><script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.14.3/dist/browser/index.min.js"></script></head><body><svg id="mindmap"></svg><script>
        const markdown = `{markdown_content}`;
        const transformer = new markmap.Transformer();
        const {{ root }} = transformer.transform(markdown);
        const mm = new markmap.Markmap(document.querySelector('#mindmap'), {{ autoFit: true, initialExpandLevel: 2 }});
        mm.setData(root);
        mm.fit();
    </script></body></html>
    """
    return html

@st.cache_resource(show_spinner="Setting up Q&A system...")
def setup_rag_system(text):
    """Sets up the Retrieval Augmented Generation (RAG) system."""
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=c) for c in splitter.split_text(text)]
    vectorstore = Chroma.from_documents(
        documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    template = """
    You are an expert study assistant. Your goal is to answer the user's question 
    accurately and concisely based ONLY on the provided context. 
    The context is a set of study notes, possibly from an exam paper and the uploaded pdf.
    If the context does not contain the answer, state that you cannot find the answer 
    in the provided documents, but try to answer based on general knowledge if possible, 
    making it clear you are going beyond the document.
    
    ---
    CONTEXT:
    {context}
    ---
    
    QUESTION: {question}
    
    ANSWER:
    """
    custom_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    def format_docs(docs: List[Document]) -> str: return "\n\n".join(doc.page_content for doc in docs)
    def create_input_dict(question: str) -> dict:
        retrieved_docs = retriever.invoke(question) 
        return {"context": format_docs(retrieved_docs), "question": question}

    return RunnableLambda(create_input_dict) | custom_prompt | llm

# -----------------------------
# 🧠 Interactive Quiz Generation & Logic
# -----------------------------

def initialize_quiz_state():
    """Initializes quiz-related session state variables."""
    if 'quiz_questions' not in st.session_state: st.session_state['quiz_questions'] = []
    if 'quiz_submitted' not in st.session_state: st.session_state['quiz_submitted'] = False
    if 'user_answers' not in st.session_state: st.session_state['user_answers'] = {}
    if 'flashcards' not in st.session_state: st.session_state['flashcards'] = []
    # New state for interactive flashcards
    if 'current_card_index' not in st.session_state: st.session_state['current_card_index'] = 0
    if 'reveal_answer' not in st.session_state: st.session_state['reveal_answer'] = False
    # New state for document text cache
    if 'document_text_cache' not in st.session_state: st.session_state['document_text_cache'] = {}
    if 'selected_tool_context' not in st.session_state: st.session_state['selected_tool_context'] = "All Combined"


def parse_json_output(json_string: str) -> List[Dict[str, Any]]:
    """Parses the JSON string from the Gemini model, cleaning up markdown fences."""
    try:
        if json_string.startswith("```json"):
            json_string = json_string.strip()[7:-3].strip()
        elif json_string.startswith("```"):
             json_string = json_string.strip()[3:-3].strip()
        
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON output: {e}")
        st.text_area("Raw JSON Output (for debugging):", json_string, height=100)
        return []

def generate_and_parse_quiz(combined_text: str, num_questions: int):
    """Calls Gemini to generate the quiz in a structured JSON format."""
    
    quiz_text_input = combined_text[:15000]

    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {
                    "type": "array", "items": {"type": "string"}, 
                    "description": "Exactly four options for the MCQ."
                },
                "correct_answer_index": {
                    "type": "integer",
                    "description": "The 0-indexed position of the correct answer (0 for A, 1 for B, etc.)"
                }
            },
            "required": ["question", "options", "correct_answer_index"]
        }
    }
    
    prompt = f"""
    Generate exactly {num_questions} Multiple-Choice Questions (MCQs) based ONLY on the provided text.
    You MUST respond with a JSON array that strictly adheres to the following schema.
    JSON SCHEMA: {json.dumps(json_schema)}
    
    Text for Quiz Generation:
    {quiz_text_input}
    
    Response MUST start with '```json' and end with '```'.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    
    quiz_data = parse_json_output(response.text)
    
    if quiz_data:
        st.session_state['quiz_questions'] = quiz_data
        st.session_state['quiz_submitted'] = False
        st.session_state['user_answers'] = {}
        st.success(f"Quiz with {len(quiz_data)} questions generated successfully! Start your test below.")
    else:
        st.warning("Quiz generation failed or returned empty data.")
        st.session_state['quiz_questions'] = []
        st.session_state['quiz_submitted'] = False


def display_quiz_and_collect_answers():
    """Renders the quiz interface and collects answers in a form."""
    questions = st.session_state['quiz_questions']
    
    st.markdown("### 📝 Your Test")
    
    # Store the form submission inside a function to handle form logic on button click
    def handle_submit():
        if len(st.session_state['user_answers']) < len(questions):
            st.error("Please answer all questions before submitting.")
        else:
            st.session_state['quiz_submitted'] = True
            st.success("Answers submitted!")
            st.rerun() # Rerun to display results

    # Create a form to wrap the entire quiz submission
    with st.form(key='quiz_form'):
        
        # Display each question
        for i, q in enumerate(questions):
            key = f"q_{i}"
            
            # Options array: [Option A, Option B, Option C, Option D]
            options_labels = [f"({chr(65+j)}) {opt}" for j, opt in enumerate(q['options'])]
            
            st.markdown(f"**{i+1}.** {q['question']}")
            
            # Use st.radio without an index to force the user to select one
            user_selection = st.radio(
                label="Choose one:", 
                options=options_labels, 
                index=None, # Crucial: Starts unselected
                key=key, 
                label_visibility="collapsed"
            )
            
            # Store the current selection (if made) directly into the user_answers dict
            if user_selection is not None:
                st.session_state['user_answers'][key] = user_selection
            
            st.markdown("---")
        
        # Submission button tied to the form's logic
        st.form_submit_button("Submit Answers & View Results", on_click=handle_submit)


def display_quiz_results():
    """Displays the user's score and feedback after submission."""
    questions = st.session_state['quiz_questions']
    user_answers = st.session_state['user_answers']
    
    if not questions or not st.session_state['quiz_submitted']:
        return

    score = 0
    total = len(questions)
    
    st.subheader("🎉 Quiz Results")
    
    for i, q in enumerate(questions):
        key = f"q_{i}"
        
        # Find the correct option label (e.g., (B) Option Text)
        correct_index = q['correct_answer_index']
        correct_option_text = q['options'][correct_index]
        correct_label = f"({chr(65+correct_index)}) {correct_option_text}"
        
        # Get the user's selected label
        user_label = user_answers.get(key, "Not Answered") 
        
        # Determine if correct
        is_correct = (user_label == correct_label)
        if is_correct:
            score += 1
            icon = "✅"
            color = "green"
        else:
            icon = "❌"
            color = "red"
            
        # Display feedback
        st.markdown(f"**{i+1}.** {q['question']}")
        
        # Display user's answer
        st.markdown(f"**Your Answer:** <span style='color:{color}'>{icon} {user_label}</span>", unsafe_allow_html=True)
        
        # Display correct answer if incorrect
        if not is_correct:
            st.markdown(f"**Correct Answer:** <span style='color:green'>{correct_label}</span>", unsafe_allow_html=True)
        
        st.markdown("---")

    # Final Score Display
    st.success(f"**Final Score:** {score} / {total} ({score/total:.0%})")
    
    # Option to reset the quiz
    if st.button("Generate New Quiz", key="reset_quiz_btn"):
        initialize_quiz_state()
        st.session_state['quiz_submitted'] = False
        st.rerun()


# -----------------------------
# 🃏 Flashcard Logic
# -----------------------------

def generate_flashcards(combined_text: str, num_cards: int):
    """Calls Gemini to generate flashcards in a structured JSON format."""
    
    flashcard_text_input = combined_text[:15000]

    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The front of the flashcard, acting as the question or term."},
                "answer": {"type": "string", "description": "The back of the flashcard, containing the definition or explanation."}
            },
            "required": ["question", "answer"]
        }
    }
    
    prompt = f"""
    Generate exactly {num_cards} Question and Answer flashcards based ONLY on the provided educational text.
    You MUST respond with a JSON array that strictly adheres to the following schema.
    JSON SCHEMA: {json.dumps(json_schema)}
    
    Text for Flashcard Generation:
    {flashcard_text_input}
    
    Response MUST start with '```json' and end with '```'.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    
    card_data = parse_json_output(response.text)
    
    if card_data:
        st.session_state['flashcards'] = card_data
        # Reset card index and state when new cards are generated
        st.session_state['current_card_index'] = 0
        st.session_state['reveal_answer'] = False
        st.success(f"Flashcards with {len(card_data)} cards generated successfully! Start studying below.")
    else:
        st.warning("Flashcard generation failed or returned empty data.")
        st.session_state['flashcards'] = []

def display_flashcards_interactive():
    """Renders a single, interactive flashcard with next/previous/flip functionality."""
    
    cards = st.session_state.get('flashcards', [])
    total_cards = len(cards)
    
    if not cards:
        st.info("Click 'Generate Flashcards' to start memorizing your notes.")
        return

    # Use callbacks to handle button clicks and state updates
    def next_card():
        st.session_state['current_card_index'] = (st.session_state['current_card_index'] + 1) % total_cards
        st.session_state['reveal_answer'] = False # Hide answer on navigation

    def prev_card():
        st.session_state['current_card_index'] = (st.session_state['current_card_index'] - 1 + total_cards) % total_cards
        st.session_state['reveal_answer'] = False

    def flip_card():
        st.session_state['reveal_answer'] = not st.session_state['reveal_answer']

    # Display current card info
    current_index = st.session_state.get('current_card_index', 0)
    
    # Bounds check (in case cards list changed size)
    if current_index >= total_cards:
        st.session_state['current_card_index'] = 0
        current_index = 0
        if total_cards == 0:
             return
             
    current_card = cards[current_index]

    st.markdown(f"**Card {current_index + 1} of {total_cards}**")
    
    # --- Rectangular Card Frame (styled with markdown/HTML) ---
    card_content = ""
    if st.session_state['reveal_answer']:
        # Show the back of the card (Answer)
        card_content = f"<div style='padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; min-height: 200px; background-color: #e8f5e9; text-align: center;'>"\
                       f"<h3>✅ Answer</h3>"\
                       f"<p style='font-size: 16px; font-weight: normal;'>{current_card['answer']}</p>"\
                       f"</div>"
    else:
        # Show the front of the card (Question)
        card_content = f"<div style='padding: 20px; border: 2px solid #2196F3; border-radius: 10px; min-height: 200px; background-color: #e3f2fd; text-align: center;'>"\
                       f"<h3>❓ Question</h3>"\
                       f"<p style='font-size: 18px; font-weight: bold;'>{current_card['question']}</p>"\
                       f"</div>"
                       
    st.markdown(card_content, unsafe_allow_html=True)
    st.markdown("---") # Separator for controls

    # --- Controls (Buttons) ---
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col1:
        st.button("⬅️ Previous", on_click=prev_card, disabled=(total_cards <= 1), use_container_width=True)
    
    with col2:
        # The Flip button
        st.button(
            "🔄 Flip Card", 
            on_click=flip_card, 
            key="flip_card_btn", 
            type="primary", 
            use_container_width=True
        )

    with col3:
        st.button("Next ➡️", on_click=next_card, disabled=(total_cards <= 1), use_container_width=True)
    

# -----------------------------
# 🚀 Main Streamlit App 
# -----------------------------
def main():
    # --- Renamed App Title ---
    st.set_page_config(page_title="LearnOverse: AI-fying your Learning Experience", layout="wide", page_icon="💡")
    st.title(" 🌠 LearnOverse(Beta) — Expanding Your Study Universe with AI")
    st.markdown("""
### Welcome to **Aiverse Learning** 🌌  
Your personal **AI-powered study companion** — upload notes, generate mindmaps, chat with your documents,  
and create quizzes & flashcards instantly.  
""")

    

    if not configure_genai():
        return
    
    initialize_quiz_state() # Initialize quiz/flashcard state variables

    # --- Inject Custom CSS for Scrolling (Using Native Container Height) ---
    st.markdown("""
        <style>
            /* Apply fixed height and scrolling to the chat/tools container in the middle */
            #middle-scroll-container {
                max-height: 650px; /* Fixed height for middle column content */
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)
    # ---------------------------------------

    left, middle, right = st.columns([1.3, 2, 1.3])
    
    # --- Initialize other session state variables ---
    if "messages" not in st.session_state: st.session_state["messages"] = [{"role": "assistant", "content": "Hello!..."}]
    if 'selected_resource_name' not in st.session_state: st.session_state['selected_resource_name'] = "" 
    if 'file_data_list' not in st.session_state: st.session_state['file_data_list'] = [] 
    if 'rag_chain' not in st.session_state: st.session_state['rag_chain'] = None 


    # LEFT COLUMN: File Input and Options (Fixed Height Container)
    with left:
        # Use a fixed-height container here to keep the left column size consistent
        with st.container(height=650, border=False):
            st.header("📂 Upload or Choose Notes")
            
            # 1. Multi-File Upload
            uploaded_files: Optional[List[Any]] = st.file_uploader(
                "📘 1. Upload one or more PDFs", 
                type=["pdf"], 
                key="main_upload",
                accept_multiple_files=True 
            )

            # Handle local file upload
            if uploaded_files:
                if not all(isinstance(f, BytesIO) for f in st.session_state.get('file_data_list', [])) or len(st.session_state.get('file_data_list', [])) != len(uploaded_files):
                    st.session_state['file_data_list'] = uploaded_files
                    st.session_state['selected_resource_name'] = f"{len(uploaded_files)} local file(s)"
                    st.info(f"Loaded: **{len(uploaded_files)}** local PDFs.")
            
            st.divider()
            
            # 2. Choose from Presets
            st.subheader("📚 2. Choose from Existing Notes")
            
            selected_link_from_presets: Optional[str] = None
            preset_name: str = ""
            
            for category_name, files in PRESET_RESOURCES.items():
                with st.container(border=True): 
                    st.markdown(f"**{category_name}**")
                    file_names = ["-- Select File --"] + list(files.keys())
                    
                    selection = st.selectbox(
                        f"Choose file from {category_name}",
                        file_names,
                        key=f"selectbox_{category_name.replace(' ', '_')}" 
                    )

                    if selection != "-- Select File --":
                        selected_link_from_presets = files[selection]
                        preset_name = selection
                        uploaded_files = None 
                        break 

            # Handle preset selection logic
            if selected_link_from_presets:
                if not st.session_state['file_data_list'] or st.session_state['selected_resource_name'] != preset_name:
                    with st.spinner(f"Downloading preset: {preset_name}..."):
                        drive_file_buffer = download_from_drive(selected_link_from_presets)
                        if drive_file_buffer:
                            st.session_state['file_data_list'] = [drive_file_buffer]
                            st.session_state['selected_resource_name'] = preset_name
                            st.success(f"Selected preset: **{preset_name}**")
                        else:
                            st.session_state['file_data_list'] = []
                            st.session_state['selected_resource_name'] = ""
                            
            
            # 3. Handle Manual Link Input 
            st.divider()
            manual_drive_link = st.text_input("🔗 Or, paste a Google Drive Link manually:")
            if manual_drive_link and ('Manual Link' not in st.session_state.get('selected_resource_name', '')):
                 with st.spinner("Downloading manual link..."):
                    drive_file_buffer = download_from_drive(manual_drive_link)
                    if drive_file_buffer:
                        st.session_state['file_data_list'] = [drive_file_buffer]
                        st.session_state['selected_resource_name'] = "Manual Link"
                        st.success("Manual link loaded.")
                    else:
                        st.session_state['file_data_list'] = []
                        st.session_state['selected_resource_name'] = ""


    # --- File Loading and Extraction Logic (Run once per change) ---
    combined_text = ""
    file_list_for_preview = [] 
    
    # Process files if they are available
    if st.session_state['file_data_list']:
        # Use a unique key for the documents currently loaded (based on file names or IDs)
        current_file_keys = tuple(f.name if hasattr(f, 'name') else f"drive_file_{i}" for i, f in enumerate(st.session_state['file_data_list']))
        
        if st.session_state.get('last_file_keys') != current_file_keys:
            
            st.session_state['document_text_cache'] = {} # Clear cache
            st.session_state['last_file_keys'] = current_file_keys
            st.session_state['selected_tool_context'] = "All Combined" # Reset context selector
            
            with st.spinner(f"Extracting text from {st.session_state['selected_resource_name']}..."):
                
                total_text_length = 0
                temp_combined_text = ""
                temp_file_list_for_preview = []
                
                for i, file_data in enumerate(st.session_state['file_data_list']):
                    try: file_data.seek(0)
                    except AttributeError: pass 

                    text = extract_text_from_pdf(file_data)
                    
                    if text:
                        file_key = current_file_keys[i]
                        # For display purposes, use a user-friendly name if available
                        display_name = file_data.name if hasattr(file_data, 'name') and file_data.name else file_key
                        st.session_state['document_text_cache'][display_name] = text
                        temp_combined_text += text + "\n\n---\n\n" 
                        total_text_length += len(text)
                        temp_file_list_for_preview.append(file_data) 

                if total_text_length > 0:
                    combined_text = temp_combined_text
                    st.session_state['combined_text'] = combined_text
                    st.session_state['file_list_for_preview'] = temp_file_list_for_preview
                    st.info(f"Total extracted text length: **{total_text_length}** characters from **{len(temp_file_list_for_preview)}** file(s).")
                else:
                    st.session_state['file_data_list'] = [] 
                    st.session_state['combined_text'] = ""
                    st.session_state['document_text_cache'] = {}
                    st.session_state['file_list_for_preview'] = []
                    st.warning("No readable text found in the selected PDF(s).")
        
        else:
             # Use cached data if resource hasn't changed
             combined_text = st.session_state.get('combined_text', "")
             file_list_for_preview = st.session_state.get('file_list_for_preview', [])

    
    # MIDDLE COLUMN: Tools (SCROLLABLE CONTAINER)
    with middle:
        # Wrap the tools content in a container to enforce scrolling
        with st.container(height=650, border=False): 
        
            st.header("🧰 Study Tools")
            
            if combined_text.strip():
                
                # --- NEW: DOCUMENT CONTEXT SELECTOR ---
                cache = st.session_state.get('document_text_cache', {})
                # Use display names from the cache keys
                options_list = ["All Combined"] + list(cache.keys())
                
                # Check if the currently selected context is still valid
                if st.session_state.get('selected_tool_context') not in options_list:
                    st.session_state['selected_tool_context'] = "All Combined"
                    
                selected_context_key = st.selectbox(
                    "🎯 **Select Content Context for Tools:**",
                    options_list,
                    key='context_selector',
                    index=options_list.index(st.session_state['selected_tool_context'])
                )
                
                # Update the context text used by ALL tools below
                if selected_context_key == "All Combined":
                    context_text = st.session_state['combined_text']
                else:
                    context_text = cache.get(selected_context_key, st.session_state['combined_text'])
                
                # Set the current context key for persistence
                st.session_state['selected_tool_context'] = selected_context_key
                st.divider()
                # ----------------------------------------
                
                # We use context_text for all downstream operations
                markdown_content = create_mindmap_markdown(context_text)
                
                if markdown_content:
                    # RAG setup should only happen if the context has changed
                    if st.session_state.get('last_context_for_rag') != selected_context_key:
                        rag_chain = setup_rag_system(context_text)
                        st.session_state['rag_chain'] = rag_chain
                        st.session_state['last_context_for_rag'] = selected_context_key
                        # Clear old chat messages to avoid confusion with new RAG source
                        st.session_state["messages"] = [{"role": "assistant", "content": f"Context changed to **{selected_context_key}**. Ask a new question based on this content."}]
                        
                    
                    # --- TABS ---
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                         
                        "💬 Chat", 
                        "📊 Mindmap",
                        "🧠 **Quiz**", 
                        "🃏 **Flashcards**",
                        "💡 Summary",
                        "📈 Report",
                        "🔊 Audio Summary",
                        "🎬 Video Summary"
                    ])

                    # Tab 1: Mindmap Viewer
                    with tab2:
                        st.subheader("Interactive Mindmap")
                        html = create_markmap_html(markdown_content)
                        components.html(html, height=450, scrolling=False) 
                        
                        encoded_html = urllib.parse.quote(html)
                        st.markdown(
                            f'<a href="data:text/html;charset=utf-8,{encoded_html}" target="_blank">'
                            f'<button style="padding:8px 16px;font-size:16px;">🌐 Open Mindmap in New Tab</button></a>',
                            unsafe_allow_html=True,
                        )

                    # Tab 2: Chat with Notes
                    with tab1:
                        st.subheader("💬 Chat with Document(s): Your Sahayak")
                        
                        for msg in st.session_state.messages:
                            st.chat_message(msg["role"]).write(msg["content"])

                        if query := st.chat_input(f"Ask about {selected_context_key}...", key="chat_input_tab4"):
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.chat_message("user").write(query)

                            with st.spinner("Thinking..."):
                                try:
                                    rag_chain = st.session_state['rag_chain']
                                    if rag_chain:
                                        result_runnable = rag_chain.invoke(query) 
                                        answer = result_runnable.content.strip() 
                                        
                                        if not answer:
                                            answer = f'I could not find a relevant answer in the documents related to **{selected_context_key}**.'
                                            
                                        st.session_state.messages.append({"role": "assistant", "content": answer})
                                        st.chat_message("assistant").write(answer)
                                    else:
                                        st.warning("RAG system not set up. Please ensure a document is successfully loaded.")
                                
                                except Exception as e:
                                    error_message = f"An error occurred during RAG processing: {e}"
                                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                                    st.chat_message("assistant").write(error_message)

                    # Tab 3: Quiz Generator (INTERACTIVE)
                    with tab3:
                        st.subheader("🧠 Interactive Quiz (MCQs)")
                        num_questions = st.slider("Number of Questions:", min_value=3, max_value=10, value=5, key="num_questions_slider")
                        
                        if st.button(f"Generate New Quiz for {selected_context_key}", key="generate_quiz_btn"):
                            with st.spinner(f"Generating {num_questions} questions..."):
                                generate_and_parse_quiz(context_text, num_questions) # Uses context_text

                        if st.session_state['quiz_questions']:
                            if not st.session_state['quiz_submitted']:
                                display_quiz_and_collect_answers()
                            else:
                                display_quiz_results()
                        else:
                            st.info(f"Click the button above to generate a quiz from **{selected_context_key}**.")
                    
                    # Tab 4: Flashcard Generator (UPDATED INTERACTIVE FEATURE)
                    with tab4:
                        st.subheader("🃏 Interactive Flashcards (Flip & Swipe)")
                        
                        with st.expander(f"Generate New Set of Flashcards from {selected_context_key}", expanded=(len(st.session_state['flashcards']) == 0)):
                            num_cards = st.slider("Number of Flashcards:", min_value=5, max_value=20, value=10, key="num_cards_slider")
                            
                            if st.button(f"Generate Flashcards", key="generate_flashcards_btn", use_container_width=True):
                                 with st.spinner(f"Generating {num_cards} cards..."):
                                     generate_flashcards(context_text, num_cards) # Uses context_text

                        st.markdown("---")
                        
                        display_flashcards_interactive()
                        
                    # Tab 5: Summary and Customization
                    with tab5:
                        st.subheader("💡 Document Summary Generator")
                        
                        st.markdown("### ⚙️ Customization Options (Context: " + selected_context_key + ")")
                        style = st.selectbox("🧪 Style", ["Conceptual", "Mathematical", "Coding", "Bullet Points", "Paragraph-wise"], key="mid_summary_style_select")
                        depth = st.selectbox("🧠 Depth", ["Intermediate", "Friendly", "Basic", "Advanced"], key="mid_summary_depth_select")
                        length = st.selectbox("📏 Length", ["Medium", "Short", "Long"], key="mid_summary_length_select")

                        st.divider()

                        if st.button(f"✨ Generate Summary for {selected_context_key}", key="generate_summary_mid_tab"):
                            with st.spinner("Summarizing..."):
                                model = genai.GenerativeModel("gemini-2.5-flash")
                                summary_text = context_text[:10000] # Use context_text
                                
                                prompt = f"""
                                Summarize the following educational text clearly and concisely, focusing on key concepts and structure.
                                Style: {style}
                                Depth: {depth}
                                Length: {length}

                                Text to Summarize:
                                {summary_text}
                                """
                                
                                summary = model.generate_content(prompt).text
                                st.subheader("📝 Generated Summary")
                                st.write(summary)
                        else:
                            st.info("Click 'Generate Summary' to produce your customized overview.")


                    # Tab 6: Generate Report
                    with tab6:
                        st.subheader("📈 In-Depth Structured Report")
                        report_style = st.selectbox("Report Focus:", ["Detailed Structure & Analysis", "Critical Review & Gaps", "Executive Summary"], key="report_style_select")
                        
                        if st.button(f"Generate Report for {selected_context_key}", key="generate_report_btn"):
                            with st.spinner("Generating detailed report..."):
                                model = genai.GenerativeModel("gemini-2.5-flash")
                                report_text_input = context_text[:20000] # Use context_text
                                
                                prompt = f"""
                                Analyze the following educational text and create a comprehensive, well-formatted markdown report based on the '{report_style}' style.
                                
                                Report Requirements:
                                1. Use appropriate markdown headings (#, ##) and bolding for structure.
                                2. Include a Table of Contents (TOC) at the top.
                                3. **Detailed Structure & Analysis:** Break down the core concepts, principles, and key facts.
                                4. **Critical Review & Gaps:** Identify assumptions, weaknesses, missing information, or complex areas needing further study.
                                5. **Executive Summary:** Provide a concise, high-level summary suitable for a manager or quick reader.

                                Text to Analyze:
                                {report_text_input}
                                """
                                report = model.generate_content(prompt).text
                                st.subheader(f"📄 Generated Report: {report_style}")
                                st.markdown(report)
                    
                    # Tab 7: Audio Summary (PLACEHOLDER)
                    with tab7:
                        st.subheader("🔊 Generate Audio Summary")
                        st.warning("This feature is **Under Construction**. Coming soon!")
                        st.markdown(f"The audio summary will use the content from **{selected_context_key}**.")
                        
                    # Tab 8: Video Summary (PLACEHOLDER)
                    with tab8:
                        st.subheader("🎬 Generate Video/Conceptual Summary")
                        st.warning("This feature is **Under Construction**. Coming soon!")
                        st.markdown(f"The video summary script will use the content from **{selected_context_key}**.")


                else:
                    st.warning("Could not generate Mindmap. Please check the document content.")
            else:
                st.info("Upload your PDFs or select a resource from the left panel to begin. No document currently loaded.")
                

    # RIGHT COLUMN: Document Viewer (Fixed Height Container)
    with right:
        # Use a fixed-height container here to keep the right column size consistent
        with st.container(height=650, border=False): 
            st.header(f"📄 Document Viewer ({len(file_list_for_preview)} loaded)")

            if file_list_for_preview:
                
                # --- Document Selector for Preview ---
                # Use the file's .name property if available, otherwise use a generic label
                preview_options = [f.name if hasattr(f, 'name') else f"File {i+1} (Drive/Manual)" 
                                   for i, f in enumerate(file_list_for_preview)]
                
                selected_file_index = st.selectbox(
                    "Select document to preview:",
                    range(len(file_list_for_preview)),
                    format_func=lambda i: preview_options[i]
                )
                
                pdf_source = file_list_for_preview[selected_file_index]
                
                try:
                    pdf_source.seek(0)
                except AttributeError:
                    pass 
                
                st.subheader(f"📘 Preview: {preview_options[selected_file_index]}")
                
                pdf_bytes = pdf_source.getvalue()
                
                if pdf_bytes:
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    # Use a fixed height for the iframe
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>' 
                    st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    st.info("Could not load PDF content for preview.")
            else:
                st.info("Upload or load PDFs to view them here.")


    # -----------------------------
    # 🌟 CORRECT LOCATION FOR FOOTER (AFTER ALL COLUMNS) 🌟
    # -----------------------------
    st.markdown("---")
    st.markdown("<h4 style='text-align: center; color: #888888;'>Made with ❤️ for PMEC Students...</h4>", unsafe_allow_html=True)

# -----------------------------
# 🏁 Run
# -----------------------------
if __name__ == "__main__":
    main()