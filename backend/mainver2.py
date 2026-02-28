from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import uuid
from groq import Groq
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing; later set to your Replit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# In-memory session storage
SESSIONS = {}

# Rubric for domain-knowledge questions across:
# - Technical Engineering
# - Business & Management
# - Government & Public Policy
DOMAIN_RUBRIC = {
    "scale": {"min": 1, "max": 5},
    "dimensions": {
        "conceptual_accuracy": {
            "label": "Conceptual Accuracy",
            "description": "Correctness of key concepts, facts, and principles relevant to the domain.",
            "criteria": {
                1: "Mostly incorrect or contains major misconceptions.",
                2: "Partially correct but with notable conceptual errors.",
                3: "Generally correct with minor inaccuracies.",
                4: "Accurate and reliable understanding of core concepts.",
                5: "Highly accurate, nuanced, and consistently precise.",
            },
        },
        "analytical_depth": {
            "label": "Analytical Depth",
            "description": "Quality of reasoning, structure of argument, and ability to examine complexity.",
            "criteria": {
                1: "Superficial reasoning with little analysis.",
                2: "Limited analysis; misses key relationships.",
                3: "Adequate analysis with some structured reasoning.",
                4: "Strong analysis; addresses trade-offs and implications.",
                5: "Exceptional depth; integrates multiple perspectives rigorously.",
            },
        },
        "application_ability": {
            "label": "Application Ability",
            "description": "Ability to apply knowledge to practical scenarios, decisions, or problem-solving.",
            "criteria": {
                1: "Unable to apply concepts to realistic situations.",
                2: "Application is weak or inconsistent.",
                3: "Applies concepts to common scenarios adequately.",
                4: "Applies concepts effectively to varied contexts.",
                5: "Applies concepts creatively and robustly to complex cases.",
            },
        },
        "context_awareness": {
            "label": "Context Awareness",
            "description": "Recognition of domain context, constraints, stakeholders, and environment.",
            "criteria": {
                1: "Ignores context or critical constraints.",
                2: "Limited awareness of relevant context.",
                3: "Acknowledges key contextual factors.",
                4: "Incorporates context and constraints well.",
                5: "Demonstrates sophisticated situational judgment and context fit.",
            },
        },
        "communication_clarity": {
            "label": "Communication Clarity",
            "description": "Clarity, organization, and coherence of explanation.",
            "criteria": {
                1: "Unclear and difficult to follow.",
                2: "Partially clear but disorganized.",
                3: "Mostly clear with acceptable structure.",
                4: "Clear, organized, and easy to follow.",
                5: "Exceptionally clear, concise, and well-structured.",
            },
        },
    },
}

# Rubric for behavioral questions using STAR framing.
BEHAVIORAL_RUBRIC = {
    "framework": "STAR",
    "scale": {"min": 1, "max": 5},
    "dimensions": {
        "situation_clarity": {
            "label": "Situation clarity",
            "description": "How clearly the candidate establishes the context and background.",
            "criteria": {
                1: "Situation is vague or missing.",
                2: "Basic context given but unclear.",
                3: "Situation is understandable with moderate detail.",
                4: "Clear and relevant context with good framing.",
                5: "Highly clear, concise, and context-rich setup.",
            },
        },
        "task_ownership": {
            "label": "Task ownership",
            "description": "How clearly the candidate defines their responsibility and role.",
            "criteria": {
                1: "Role/responsibility is not clear.",
                2: "Limited ownership; role is ambiguous.",
                3: "Role is identified with acceptable clarity.",
                4: "Strong ownership and clear responsibility.",
                5: "Explicit ownership with strategic understanding of responsibility.",
            },
        },
        "action_specificity": {
            "label": "Action specificity",
            "description": "Specificity and relevance of actions taken by the candidate.",
            "criteria": {
                1: "Actions are absent or generic.",
                2: "Few specifics; actions loosely connected to task.",
                3: "Reasonably specific actions with logical flow.",
                4: "Detailed, targeted actions that address the task well.",
                5: "Highly specific, intentional, and well-prioritized actions.",
            },
        },
        "result_measurability": {
            "label": "Result measurability",
            "description": "Clarity and quantifiability of outcomes and impact.",
            "criteria": {
                1: "No clear result provided.",
                2: "Result stated but not measurable.",
                3: "Result is clear with limited metrics.",
                4: "Result includes meaningful metrics or outcomes.",
                5: "Result is clearly measurable with strong, relevant evidence of impact.",
            },
        },
        "reflection_learning": {
            "label": "Reflection & learning",
            "description": "Depth of reflection, lessons learned, and improvement mindset.",
            "criteria": {
                1: "No reflection or learning identified.",
                2: "Minimal reflection with generic takeaway.",
                3: "Some reflection and practical learning.",
                4: "Thoughtful reflection with clear lesson transfer.",
                5: "Deep reflection demonstrating growth and future application.",
            },
        },
    },
}
import json

def evaluate_answer_with_llm(question, answer, question_type, category):
    rubric_context = retrieve_rubric_context(
         question + " " + answer,
         question_type
)
    
    

    prompt = f"""
You are an expert interview evaluator.

Category: {category}
Question: {question}
Candidate Answer: {answer}

Use the following evaluation dimensions strictly.

Retrieved Evaluation Criteria:
{rubric_context}

Score each dimension from 1 to 5.

Return ONLY valid JSON in this format:

{{
  "dimension_scores": {{
    "dimension_name": score
  }},
  "average_score": number
}}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a strict and structured evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content

    try:
        evaluation = json.loads(content)
    except:
        # fallback safety in case model adds extra text
        content = content[content.find("{"):content.rfind("}")+1]
        evaluation = json.loads(content)

    return evaluation

def generate_follow_up_question(previous_question, previous_answer, category):
    prompt = f"""
You are an interviewer for a mid-level {category} role (2 to 4 years experience).

The candidate was asked:

Question:
{previous_question}

Their answer was:
{previous_answer}

The answer was weak or incomplete.

Generate ONE follow-up question that:
- Focuses only on the SAME specific concept
- Is clear and concise (max 1 to 2 sentences)
- Is easier or moderately difficult (not senior-architect level)
- Asks for clarification, deeper explanation, or correction
- Does NOT introduce a new topic
- Is answerable within 2 to 3 minutes

Return only the follow-up question as plain text.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a structured domain interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

def generate_domain_question(session):
    category = session["category"]
    resume_index = session["resume_index"]
    resume_chunks = session["resume_chunks"]

    previous_questions = [
        q["question"]
        for q in session["questions"]
        if q["question_type"] == "domain"
    ]

    # 🔥 FAISS Retrieval
    # Retrieve job knowledge context
    job_context = retrieve_relevant_context(
        category,
        category
    )

# Retrieve resume-specific context
    resume_context = retrieve_resume_context(
        category,
        resume_index,
        resume_chunks
    )

    prompt = f"""
You are an interviewer for a mid-level {category} role.
Retrieved job knowledge context:
{job_context}

Retrieved relevant resume context:
{resume_context}

Previously asked domain questions:
{previous_questions}

Generate ONE domain-specific interview question that:
- Tests applied understanding
- Is grounded in the retrieved context
- Does NOT repeat previous topics
- Is specific, not generic
- Is clear and concise (max 2 sentences)
- Focuses on ONE specific concept only
- Is moderately difficult (not senior-architect level)
- Is realistic for a 2 to 4 year experience candidate
- Is directly related to the resume context
- Does NOT combine multiple large system design parts

Return only the question.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a professional domain interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()

def generate_behavioral_question(session):
    previous_questions = [q["question"] for q in session["questions"] if q["question_type"] == "behavioral"]

    prompt = f"""
You are an expert behavioral interviewer.

Previously asked behavioral questions:
{previous_questions}

Generate ONE behavioral interview question.
Requirements:
- Must follow STAR-style evaluation logic.
- Must probe leadership, conflict, decision-making, ethics, or ownership.
- Must not repeat previously asked themes.
- Must be realistic and professional.

Return only the question text.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a structured behavioral interviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

    return response.choices[0].message.content.strip()
def generate_final_report(session):
    domain_scores = session.get("domain_scores", [])
    behavioral_scores = session.get("behavioral_scores", [])

    avg_domain = sum(domain_scores) / len(domain_scores) if domain_scores else 0
    avg_behavioral = sum(behavioral_scores) / len(behavioral_scores) if behavioral_scores else 0

    overall = (avg_domain + avg_behavioral) / 2 if (domain_scores and behavioral_scores) else avg_domain or avg_behavioral

    answers_summary = []
    for ans in session["answers"]:
        answers_summary.append({
            "question": ans["question"],
            "score": ans["evaluation"]["average_score"]
        })

    prompt = f"""
You are an expert interview coach.

Interview Category: {session["category"]}

Domain Average Score: {avg_domain}
Behavioral Average Score: {avg_behavioral}
Overall Score: {overall}

Answer Scores:
{answers_summary}

Generate:

1) Strength Summary (bullet points)
2) Weakness Summary (bullet points)
3) Improvement Roadmap:
   - Domain knowledge improvement plan
   - Behavioral (STAR) improvement plan
   - 30-day structured improvement roadmap

Be specific, professional, and constructive.
Do NOT be harsh.
Do NOT repeat raw scores.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a structured interview performance analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    report_text = response.choices[0].message.content.strip()

    return {
        "average_domain_score": round(avg_domain, 2),
        "average_behavioral_score": round(avg_behavioral, 2),
        "overall_score": round(overall, 2),
        "report": report_text
    }
class StartInterviewRequest(BaseModel):
    resume_text: str
    job_role: str


class GenerateQuestionRequest(BaseModel):
    session_id: str


class SubmitAnswerRequest(BaseModel):
    session_id: str
    answer: str


@app.get("/")
def root():
    return {"message": "Backend is alive"}


    #JOB KNOWLEDGE DICTIONARY FOR FAISS
# Master job knowledge base
JOB_KNOWLEDGE = {
    "Technical Engineering": """
    Required skills:
    - Data structures and algorithms
    - Programming proficiency (Python/Java/C++/JavaScript)
    - System design fundamentals
    - Database management (SQL/NoSQL)
    - API development and integration
    - Version control (Git)
    - Debugging and testing methodologies
    - Basic cloud knowledge (AWS/GCP/Azure)
    - Problem-solving and optimization skills
    - Software development lifecycle understanding
    """,

    "Business & Management": """
    Required skills:
    - Strategic thinking
    - Market analysis
    - Financial literacy
    - Communication and stakeholder management
    - Leadership and team coordination
    - KPI and performance evaluation
    - Problem-solving in business contexts
    - Process optimization
    - Basic data interpretation skills
    - Decision-making under uncertainty
    """,

    "Government & Public Policy": """
    Required skills:
    - Indian polity and constitution knowledge
    - Governance and public administration concepts
    - Economic and social policy understanding
    - Analytical writing
    - Ethical reasoning and integrity
    - Current affairs awareness
    - Decision-making under constraints
    - Leadership in public systems
    - Crisis management
    - Public communication skills
    """
}
def build_faiss_index():
    documents = []
    metadata = []

    for category, text in JOB_KNOWLEDGE.items():
        chunks = text.split("\n")

        for chunk in chunks:
            clean = chunk.strip()
            if clean:
                documents.append(clean)
                metadata.append(category)

    embeddings = embedding_model.encode(documents)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, documents, metadata
faiss_index, faiss_docs, faiss_meta = build_faiss_index()
# 🔥 RUBRIC FAISS INDEX STARTS HERE
# =====================================================

def build_rubric_index():
    documents = []
    metadata = []

    # DOMAIN RUBRIC
    for dim_name, dim_data in DOMAIN_RUBRIC["dimensions"].items():
        text = f"""
        Dimension: {dim_data["label"]}
        Description: {dim_data["description"]}
        Criteria: {dim_data["criteria"]}
        """
        documents.append(text.strip())
        metadata.append(("domain", dim_name))

    # BEHAVIORAL RUBRIC
    for dim_name, dim_data in BEHAVIORAL_RUBRIC["dimensions"].items():
        text = f"""
        Dimension: {dim_data["label"]}
        Description: {dim_data["description"]}
        Criteria: {dim_data["criteria"]}
        """
        documents.append(text.strip())
        metadata.append(("behavioral", dim_name))

    embeddings = embedding_model.encode(documents)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, documents, metadata


# 🔥 BUILD RUBRIC INDEX GLOBALLY
rubric_index, rubric_docs, rubric_meta = build_rubric_index()

def build_resume_index(resume_text):
    chunks = []

    # Simple chunking by sentences/lines
    for line in resume_text.split("\n"):
        clean = line.strip()
        if clean:
            chunks.append(clean)

    if not chunks:
        chunks = [resume_text]

    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks
def retrieve_resume_context(query, resume_index, resume_chunks, top_k=5):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = resume_index.search(query_embedding, top_k)

    retrieved = []
    for idx in indices[0]:
        retrieved.append(resume_chunks[idx])

    return "\n".join(retrieved)

def retrieve_relevant_context(query, category, top_k=5):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = faiss_index.search(query_embedding, top_k)

    retrieved = []

    for idx in indices[0]:
        if faiss_meta[idx] == category:
            retrieved.append(faiss_docs[idx])

    return "\n".join(retrieved)
def retrieve_rubric_context(query, question_type, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = rubric_index.search(query_embedding, top_k)

    retrieved = []

    for idx in indices[0]:
        rubric_type, dim_name = rubric_meta[idx]

        if rubric_type == question_type:
            retrieved.append(rubric_docs[idx])

    return "\n\n".join(retrieved)
# MAPPING FRONTEND job profile names to internal category
# Frontend guys pls Replace the placeholder strings with exact frontend labels

JOB_PROFILE_MAPPING = {

    # -------- Technical Engineering --------
    "Software Engineer": "Technical Engineering",
    "Data Scientist": "Technical Engineering",

    # -------- Business & Management --------
    "Business Analyst": "Business & Management",
    "Product Manager": "Business & Management",

    # -------- Government & Public Policy --------
    "Civil Services": "Government & Public Policy",
    "Policy and Planning Officer": "Government & Public Policy",
}

#Interview session initialisation
@app.post("/start-interview")
def start_interview(data: StartInterviewRequest):
    session_id = str(uuid.uuid4())

    category = JOB_PROFILE_MAPPING.get(
        data.job_role,
        "Technical Engineering"  # fallback default
    )
    #Build resume FAISS index
    resume_index, resume_chunks = build_resume_index(data.resume_text)

    SESSIONS[session_id] = {
        "resume_text": data.resume_text,
        "resume_index": resume_index,
        "resume_chunks": resume_chunks,
        "job_role": data.job_role,
        "category": category,
        "questions": [],
        "answers": [],
        "domain_scores": [],
        "behavioral_scores": [],
        "question_count": 0,
        "max_questions": 10,
        "follow_up_depth": 0,
    }

    return {
        "session_id": session_id,
        "resolved_category": category
    }


@app.post("/generate-question")
def generate_question(data: GenerateQuestionRequest):
    session = SESSIONS.get(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    questions = session["questions"]

    # Determine next question type
    if questions:
        last_question_type = questions[-1]["question_type"]
        next_question_type = "behavioral" if last_question_type == "domain" else "domain"
    else:
        next_question_type = "domain"

    # Placeholder question generation
    if next_question_type == "domain":
        question = generate_domain_question(session)
    else:
        question = generate_behavioral_question(session)
    

    # Store question in session
    questions.append({
        "question": question,
        "question_type": next_question_type,
    })

    return {
        "question": question,
        "question_type": next_question_type,
    }


@app.post("/submit-answer")
def submit_answer(data: SubmitAnswerRequest):
    session = SESSIONS.get(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    questions = session.get("questions", [])
    if not questions:
        raise HTTPException(status_code=400, detail="No question has been generated yet")

    last_question = questions[-1]
    question_type = last_question.get("question_type")
    if question_type not in {"domain", "behavioral"}:
        raise HTTPException(status_code=400, detail="Invalid last question type")

    session["question_count"] += 1

    # --- Evaluate using LLM ---
    evaluation = evaluate_answer_with_llm(
        last_question.get("question"),
        data.answer,
        question_type,
        session["category"]
    )

    try:
        score = float(evaluation["average_score"])
    except (ValueError, TypeError):
        score = 3.0
    # Store answer record
    session["answers"].append({
        "question": last_question.get("question"),
        "question_type": question_type,
        "answer": data.answer,
        "evaluation": evaluation
    })

    # Track scores
    if question_type == "domain":
        session["domain_scores"].append(score)
    else:
        session["behavioral_scores"].append(score)

    # --- Interview Completion Check (ALWAYS CHECK FIRST) ---
    if session["question_count"] >= session.get("max_questions", 10):
        final_report = generate_final_report(session)

        return {
        "message": "Interview complete.",
        "interview_complete": True,
        "session_id": data.session_id,
        "final_evaluation": final_report
    }

    # --- Follow-up Logic ---
    follow_up_depth = session.get("follow_up_depth", 0)

    if score <= 2 and follow_up_depth < 1:
        next_question_type = question_type
        session["follow_up_depth"] = follow_up_depth + 1

        next_question = generate_follow_up_question(
            last_question.get("question"),
            data.answer,
            session["category"]
        )

        transition_message = "Let's explore that a bit deeper."

    else:
        session["follow_up_depth"] = 0
        next_question_type = "behavioral" if question_type == "domain" else "domain"

        if score >= 4:
            transition_message = "Good response. Moving to the next question."
        else:
            transition_message = "Thank you. Moving to the next question."

        if next_question_type == "domain":
            next_question = generate_domain_question(session)
        else:
            next_question = generate_behavioral_question(session)

    # --- Append next question ---
    session["questions"].append({
        "question": next_question,
        "question_type": next_question_type,
    })

    return {
        "message": transition_message,
        "next_question": next_question,
        "question_type": next_question_type,
        "interview_complete": False
    }
@app.get("/final-report/{session_id}")
def get_final_report(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["question_count"] < session.get("max_questions", 10):
        return {
            "message": "Interview not yet complete.",
            "interview_complete": False
        }

    report = generate_final_report(session)

    return {
        "interview_complete": True,
        "final_evaluation": report
    }