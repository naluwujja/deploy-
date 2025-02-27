import streamlit as st
import uuid
import re
try:
    import torch
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "torch>=1.13.0"])
    import torch

# Then try to import auto-gptq or install it if needed
try:
    import auto_gptq
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "auto-gptq"])
    import auto_gptq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# You'll need to install unsloth and import it properly
import unsloth
from unsloth import FastLanguageModel

# App title and description
st.set_page_config(page_title="Learning Disability Consultation System", layout="wide")
st.title("Learning Disability Consultation System")

st.markdown("""
This system will ask you questions about your symptoms and experiences related to potential learning disabilities.
At the end, it will generate a preliminary report you can share with healthcare professionals.

**Note:** This is not a clinical diagnosis but a preliminary assessment tool.
""")

# Session state initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.user_session_id = str(uuid.uuid4())
    st.session_state.user_responses_cache = []
    st.session_state.context_cache = []
    st.session_state.previous_questions = []
    st.session_state.question_count = 0
    st.session_state.question_cap = 15
    st.session_state.consultation_complete = False
    st.session_state.evaluation_report = ""
    st.session_state.current_question = "What symptoms are you experiencing? (Click 'Submit' to start)"

# Sidebar for configuration and info
with st.sidebar:
    st.header("Configuration")
    st.session_state.question_cap = st.slider("Maximum questions", 5, 20, 15)
    
    st.header("Session Info")
    st.write(f"Session ID: {st.session_state.user_session_id}")
    st.write(f"Questions asked: {st.session_state.question_count}/{st.session_state.question_cap}")
    
    if st.button("Reset Session"):
        st.session_state.user_session_id = str(uuid.uuid4())
        st.session_state.user_responses_cache = []
        st.session_state.context_cache = []
        st.session_state.previous_questions = []
        st.session_state.question_count = 0
        st.session_state.consultation_complete = False
        st.session_state.evaluation_report = ""
        st.session_state.current_question = "What symptoms are you experiencing? (Click 'Submit' to start)"
        st.experimental_rerun()

# Function to initialize the model and settings (only run once)
@st.cache_resource
def initialize_system():
    # Set up HuggingFace embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = None
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25
    
    # Configure HuggingFace login (you'll need to handle authentication securely)
    from huggingface_hub import login
    # Use st.secrets for sensitive information in production
    hf_token = st.secrets["hf_token"] if "hf_token" in st.secrets else "your_token_here"
    login(hf_token)
    
    # Load model (adjust paths for your deployment)
    model_path = 'your_model_path'  # Update with your model path
    max_seq_length = None
    dtype = None
    load_in_4bit = True
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        token = hf_token,
    )
    
    model = FastLanguageModel.for_inference(model)
    model.eval()
    
    # Load document corpus for RAG (adjust for your deployment)
    # In production, you might want to use st.cache_data for this
    try:
        with open("cleaned_text.txt", "r", encoding="utf-8") as f:
            text = f.read()
        documents = [Document(text=text)]
        index = VectorStoreIndex.from_documents(documents)
    except FileNotFoundError:
        # For demo purposes, create a small sample text if file not found
        st.warning("Document corpus not found. Using sample data.")
        sample_text = """
        ADHD (Attention-Deficit/Hyperactivity Disorder) is a neurodevelopmental disorder characterized by persistent patterns of inattention, hyperactivity, and impulsivity that interferes with functioning or development.
        
        Common symptoms include:
        - Difficulty sustaining attention in tasks or play activities
        - Not following through on instructions and failing to finish tasks
        - Easily distracted by extraneous stimuli
        - Fidgeting with hands or feet or squirming in seat
        - Difficulty waiting one's turn
        - Talking excessively and interrupting others
        
        Autism spectrum disorder (ASD) is characterized by persistent difficulties in social communication and interaction, along with restricted, repetitive patterns of behavior, interests, or activities.
        
        Dyslexia is characterized by difficulties with accurate and/or fluent word recognition and by poor spelling and decoding abilities.
        
        Dyscalculia involves difficulties in learning or comprehending arithmetic, such as difficulty understanding numbers, learning how to manipulate numbers, and learning math facts.
        
        APD (Auditory Processing Disorder) is characterized by difficulties in processing auditory information in the central nervous system, despite normal hearing abilities.
        """
        documents = [Document(text=sample_text)]
        index = VectorStoreIndex.from_documents(documents)
    
    # Initialize retriever
    top_k = 3
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )
    
    return model, tokenizer, query_engine, top_k

# Define topic areas to ensure diversity
topic_areas = [
    "attention and focus",
    "reading and writing difficulties",
    "math and numerical processing",
    "social interaction challenges",
    "sensory processing issues",
    "organization and time management",
    "memory difficulties",
    "motor skills coordination",
    "language and communication",
    "executive functioning",
    "emotional regulation",
    "anxiety or stress related to learning",
    "interests and strengths",
    "environmental factors",
    "coping strategies"
]

# Function to check for similar questions
def is_similar_to_previous(new_question, previous_questions, threshold=0.5):
    """Check if a question is semantically similar to any previous questions"""
    new_question_lower = new_question.lower().strip()
    for prev_q in previous_questions:
        prev_q_lower = prev_q.lower().strip()

        # Exact match check
        if new_question_lower == prev_q_lower:
            st.warning("EXACT DUPLICATE DETECTED!")
            return True

        # Substantial overlap check
        if len(prev_q_lower) > 10 and (prev_q_lower in new_question_lower or new_question_lower in prev_q_lower):
            st.warning("SUBSTANTIAL OVERLAP DETECTED!")
            return True

        # Key phrase detection
        key_phrases = ["inattention", "follow instructions", "stay on task", "focus",
                      "distraction", "hyperactivity", "impulsivity", "organization",
                      "reading difficulty", "writing difficulty", "math difficulty",
                      "social interaction", "sensory processing"]

        prev_phrases = [phrase for phrase in key_phrases if phrase in prev_q_lower]
        new_phrases = [phrase for phrase in key_phrases if phrase in new_question_lower]

        # If they share 2+ key phrases, consider them similar
        shared_phrases = set(prev_phrases).intersection(set(new_phrases))
        if len(shared_phrases) >= 2:
            st.warning(f"SHARED KEY PHRASES DETECTED: {shared_phrases}")
            return True

        # Core concept comparison (normalized)
        def normalize_question(q):
            q = q.lower()
            q = re.sub(r'[^\w\s]', '', q)  # Remove punctuation
            q = re.sub(r'\b(do|you|can|the|a|an|is|are|when|how|what|why|where|which|who|will|your)\b', '', q)
            return " ".join(q.split())  # Normalize whitespace

        norm_new = normalize_question(new_question)
        norm_prev = normalize_question(prev_q)

        # Calculate word overlap
        new_words = set(norm_new.split())
        prev_words = set(norm_prev.split())
        if not new_words or not prev_words:
            continue

        # Calculate Jaccard similarity
        overlap = len(new_words.intersection(prev_words))
        union = len(new_words.union(prev_words))
        similarity = overlap / union if union > 0 else 0

        if similarity > threshold:
            st.warning(f"SEMANTIC SIMILARITY DETECTED: {similarity:.2f} > {threshold}")
            return True

    return False

# Function to generate the next question
def generate_next_question(model, tokenizer, query_engine, top_k):
    # Increment question count
    st.session_state.question_count += 1
    
    # If first question, retrieve context
    if st.session_state.question_count == 1:
        try:
            # Get first response
            last_user_input = st.session_state.user_responses_cache[-1].replace("User: ", "")
            response = query_engine.query(last_user_input)
            
            # Build context
            new_context = "Context from documents:\n"
            for i in range(top_k):
                new_context += response.source_nodes[i].text + "\n\n"
            
            st.session_state.context_cache.append(new_context)
        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            st.session_state.context_cache.append("No additional context available.")
    
    # Use context from first question
    current_context = st.session_state.context_cache[0] if st.session_state.context_cache else "No context available."
    
    # Force topic rotation
    current_topic_focus = topic_areas[st.session_state.question_count % len(topic_areas)]
    
    # Format previous questions
    previous_questions_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(st.session_state.previous_questions)])
    
    # Enhanced instructions with emphasis on diversity
    instructions_string = """You are a virtual medical consultant diagnosing learning disabilities (ADHD, Autism, Dyslexia, Dyscalculia, and APD).
You engage with the user by asking relevant follow-up questions to gather more details.

### ABSOLUTELY CRITICAL INSTRUCTIONS: ###
1. YOU MUST NEVER REPEAT THE SAME QUESTION OR ASK QUESTIONS THAT ARE SIMILAR TO PREVIOUS ONES.
2. Each follow-up question MUST explore a COMPLETELY DIFFERENT aspect of learning disabilities.
3. Use the provided medical context to diversify your questions across different symptoms and experiences.
4. Before formulating a question, review the list of previous questions carefully.
5. If you notice your questions becoming repetitive, IMMEDIATELY switch to a different aspect of learning disabilities.
6. Focus on specific experiences rather than general yes/no questions.
7. Rotate between different learning disability domains (attention, reading, math, processing, social, etc.).
"""
    
    # Improved prompt structure with emphasis on diversity
    prompt = f"""[INST]
{instructions_string}

### CONTEXT (YOU MUST USE THIS TO FORM YOUR QUESTION): ###
{current_context}

User's responses so far:
{'. '.join(st.session_state.user_responses_cache)}

ALL PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT OR ASK ANYTHING SIMILAR TO THESE):
{previous_questions_formatted}

CURRENT QUESTION COUNT: {st.session_state.question_count} of {st.session_state.question_cap}

IMPORTANT DIRECTION: For this question, focus specifically on the topic of "{current_topic_focus}" which has NOT been covered yet.

TASK: Based on the CONTEXT provided, generate a NEW follow-up question about learning disability symptoms.
Your question MUST be informed by the medical context provided and relate to diagnosing learning disabilities.
The question MUST be COMPLETELY DIFFERENT from all previous questions.

FORMAT:
1. First, explicitly show your reasoning process. Start with "REASONING:" and then think step by step about:
   - What information we already have from the user
   - What diagnostic information we still need about {current_topic_focus}
   - How this topic differs from all previous questions
   - Why this question will yield new useful diagnostic information

2. Then, provide your final question after "FINAL QUESTION:" with proper capitalization and punctuation.

Both your reasoning and question will be visible to help improve the diagnostic process.
[/INST]
"""
    
    # Track attempts to generate a unique question
    max_generation_attempts = 8
    found_valid_question = False
    generation_attempts = 0
    model_response = ""
    model_reasoning = ""
    final_question = ""
    
    with st.spinner(f"Generating question {st.session_state.question_count}..."):
        while not found_valid_question and generation_attempts < max_generation_attempts:
            generation_attempts += 1
            
            try:
                # Tokenize and generate response with increased temperature to encourage diversity
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(
                    input_ids=inputs["input_ids"].to("cuda"),
                    max_new_tokens=1000,
                    temperature=0.7 + (generation_attempts * 0.1)  # Increase temperature with each retry
                )
                model_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                
                # Extract reasoning and question parts
                if "REASONING:" in model_response and "FINAL QUESTION:" in model_response:
                    parts = model_response.split("FINAL QUESTION:")
                    if len(parts) >= 2:
                        reasoning_part = parts[0]
                        question_part = parts[1].strip()
                        
                        # Extract just the reasoning without the label
                        if "REASONING:" in reasoning_part:
                            model_reasoning = reasoning_part.split("REASONING:")[1].strip()
                        else:
                            model_reasoning = reasoning_part.strip()
                        
                        final_question = question_part
                else:
                    # Fallback parsing if the model didn't follow the format exactly
                    sentences = re.split(r'(?<=[.!?])\s+', model_response)
                    question_sentences = [s for s in sentences if '?' in s]
                    
                    if question_sentences:
                        final_question = question_sentences[-1].strip()  # Take the last question
                        # Everything before is considered reasoning
                        model_reasoning = model_response[:model_response.rfind(final_question)].strip()
                    else:
                        # No question found, treat everything as reasoning
                        model_reasoning = model_response
                        final_question = f"Can you tell me about your experiences with {current_topic_focus}?"
                
                # Fix capitalization
                if final_question and final_question[0].islower():
                    final_question = final_question[0].upper() + final_question[1:]
                
                # Check for duplicate
                is_duplicate = is_similar_to_previous(final_question, st.session_state.previous_questions)
                
                if not is_duplicate:
                    found_valid_question = True
                else:
                    # Add more explicit instructions for diversity with each retry
                    prompt += f"\n\nCRITICAL ISSUE: You have generated a question that is too similar to one previously asked. Your question: '{final_question}' is similar to a previous question. Focus EXCLUSIVELY on the topic of '{current_topic_focus}' and make sure the question has NO OVERLAP in phrasing or concept with previous questions."
                    
                    # Add randomness to break out of repetition loops
                    prompt += f"\n\nSUGGESTION: Consider asking about one of these specific aspects: {', '.join(topic_areas[generation_attempts:generation_attempts+3])}"
            
            except Exception as e:
                st.error(f"Error generating response: {e}")
                continue
        
        # If we couldn't generate a unique question after multiple attempts, use a fallback strategy
        if not found_valid_question:
            st.warning("Could not generate a unique question after multiple attempts. Using fallback question.")
            
            # Create a direct question from the forced topic with no overlap
            fallback_questions = [
                f"Can you describe any challenges you experience with {current_topic_focus}?",
                f"How do difficulties with {current_topic_focus} impact your daily activities?",
                f"In what situations do you notice problems related to {current_topic_focus}?",
                f"Have others commented on your {current_topic_focus}? What did they observe?",
                f"When did you first notice issues with {current_topic_focus}?",
                f"What strategies have you tried to help with {current_topic_focus}?",
                f"How do problems with {current_topic_focus} affect your performance in different environments?",
                f"Can you compare your {current_topic_focus} abilities to those of your peers?"
            ]
            
            # Choose a fallback based on question count
            fallback_index = (st.session_state.question_count + generation_attempts) % len(fallback_questions)
            final_question = fallback_questions[fallback_index]
            model_reasoning = f"[FALLBACK REASONING]: After multiple attempts to generate a unique question, switching to a predefined question about {current_topic_focus} to ensure diversity and prevent repetition."
            
            # Double-check even the fallback for similarity
            while is_similar_to_previous(final_question, st.session_state.previous_questions) and fallback_index < len(fallback_questions) - 1:
                fallback_index += 1
                final_question = fallback_questions[fallback_index]
    
    # Add this question to the list of previous questions
    st.session_state.previous_questions.append(final_question)
    st.session_state.current_question = final_question
    
    # Return both reasoning and question
    return model_reasoning, final_question

# Function to generate evaluation report
def generate_evaluation_report(model, tokenizer):
    with st.spinner("Generating evaluation report..."):
        # Use all gathered context for the final report - using only the first question's context
        final_context = st.session_state.context_cache[0] if st.session_state.context_cache else "No context available."
        
        # Create comprehensive report prompt
        report_prompt = f"""[INST]
You are a virtual medical consultant diagnosing learning disabilities (ADHD, Autism, Dyslexia, Dyscalculia, and APD).
Based on the user's responses and the retrieved context, generate a comprehensive evaluation report.

Context from documents:
{final_context}

User's responses during the consultation:
{'. '.join(st.session_state.user_responses_cache)}

Questions asked during the consultation:
{'. '.join(st.session_state.previous_questions)}

Generate a detailed evaluation report that includes:
1. First, show your DIAGNOSTIC REASONING process:
   - Analyze the symptoms reported and their significance
   - Consider how these symptoms map to different learning disabilities
   - Discuss any patterns, contradictions, or gaps in the information
   - Evaluate the strength of evidence for each potential diagnosis

2. Then, provide the EVALUATION REPORT with:
   - Summary of the user's symptoms and reported experiences
   - Analysis of potential learning disabilities that might be present, with evidence from their responses
   - Recommended next steps for formal diagnosis
   - Disclaimer that this is an AI-generated preliminary report and not a clinical diagnosis
   - Resources that might be helpful for the user

FORMAT: Structure the report with clear headings to separate your reasoning from the final report.
TONE: Professional but accessible to someone without medical training.
[/INST]
"""
        
        try:
            # Generate the report
            inputs = tokenizer(report_prompt, return_tensors="pt")
            outputs = model.generate(
                input_ids=inputs["input_ids"].to("cuda"),
                max_new_tokens=2000,  # Allow for a comprehensive report with reasoning
                temperature=0.3  # Keep it consistent and factual
            )
            evaluation_report = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return evaluation_report
            
        except Exception as e:
            st.error(f"Error generating evaluation report: {e}")
            return "Error generating report. Please try again or consult with a healthcare professional directly."

# Main app flow
try:
    # Initialize the system (cached)
    model, tokenizer, query_engine, top_k = initialize_system()
    
    # Display the conversation history
    st.header("Consultation Session")
    
    # Create a container for chat history
    chat_container = st.container()
    
    with chat_container:
        # Display previous exchanges
        for i in range(len(st.session_state.previous_questions)):
            if i < len(st.session_state.user_responses_cache):
                st.markdown(f"**Question {i+1}:** {st.session_state.previous_questions[i]}")
                st.markdown(f"**Your response:** {st.session_state.user_responses_cache[i].replace('User: ', '')}")
                st.markdown("---")
    
    # Current question area
    st.subheader(f"Question {st.session_state.question_count + 1}/{st.session_state.question_cap}")
    st.markdown(f"**{st.session_state.current_question}**")
    
    # User response area
    user_response = st.text_area("Your response:", height=100)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("Submit")
    with col2:
        if st.button("Finish Consultation"):
            st.session_state.consultation_complete = True
    
    # When user submits a response
    if submit_button and user_response and not st.session_state.consultation_complete:
        # Store user response
        st.session_state.user_responses_cache.append(f"User: {user_response}")
        
        # Check if we've reached the question limit
        if st.session_state.question_count >= st.session_state.question_cap - 1:
            st.session_state.consultation_complete = True
        else:
            # Generate next question
            reasoning, question = generate_next_question(model, tokenizer, query_engine, top_k)
            
            # Update display (optional: can show reasoning in a collapsible section)
            with st.expander("See AI reasoning for this question"):
                st.write(reasoning)
            
            # Force refresh to show new question and previous responses
            st.experimental_rerun()
    
    # Generate and display report when consultation is complete
    if st.session_state.consultation_complete:
        st.header("Evaluation Report")
        
        if not st.session_state.evaluation_report:
            # Generate the report if it doesn't exist yet
            st.session_state.evaluation_report = generate_evaluation_report(model, tokenizer)
        
        # Display the report
        st.markdown(st.session_state.evaluation_report)
        
        # Download button for the report
        report_text = st.session_state.evaluation_report
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"learning_disability_report_{st.session_state.user_session_id}.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please refresh the page and try again. If the problem persists, contact technical support.")
