# 📚 AI Study Partner – RAG Study Assistant

An AI-powered study assistant that transforms uploaded PDF study materials into an interactive learning experience.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate grounded answers, summaries, quizzes, and flashcards.

---

## 🚀 Features

### 💬 Ask Questions
Ask questions about uploaded study materials and receive answers grounded in the document.

- Semantic document search
- AI-generated answers
- Relevant source pages displayed
- Reduced risk of unsupported responses

### 📝 Topic Summaries
Enter a topic and generate a structured study summary based on relevant sections of the uploaded document.

Summaries may include:

- Definitions
- Key concepts
- Examples
- Tables
- Time complexity
- Implementation details
- Code snippets

### 🧠 Quiz Generator
Generate multiple-choice quizzes from the study material.

The quiz system includes:

- Multiple-choice questions
- Automatic scoring
- Correct-answer feedback
- Explanations for each question

### 📁 Flashcard Generator
Create interactive flashcards for a selected topic.

Students can:

- View generated questions
- Expand individual cards
- Reveal answers when needed

---

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation pipeline:

```text
PDF Upload
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Sentence Embeddings
     ↓
FAISS Vector Index
     ↓
Semantic Retrieval
     ↓
Relevant Document Context
     ↓
Large Language Model
     ↓
Grounded Study Response

🔍 RAG Pipeline
1. PDF Processing

Uploaded PDF files are processed and readable text is extracted page by page.

The application keeps page metadata so retrieved information can be traced back to the original document.

2. Text Chunking

Extracted text is divided into smaller overlapping sections.

Current configuration:

Chunk Size: 200
Chunk Overlap: 40

The overlap helps preserve context between neighboring chunks

3. Embedding Generation

Each text chunk is converted into a numerical embedding using a Sentence Transformer model.

Embeddings represent the semantic meaning of the text rather than relying only on keyword matching.

4. FAISS Vector Search

The generated embeddings are stored in a FAISS vector index.

When a user asks a question or enters a topic:

The query is converted into an embedding.
FAISS compares it with document embeddings.
The most semantically relevant chunks are retrieved.

The application currently retrieves the top relevant sections for generation.


5. AI Generation

Retrieved document context is passed to the language model together with the user's request.

The model then generates:

Answers
Summaries
Quizzes
Flashcards

using information retrieved from the uploaded study materials.

## Tech Stack
Python | Streamlit | PyPDF | Sentence Transformers | FAISS | Large Language Model | Pandas | HTML | CSS

## Project Goals
This project was built to explore how Retrieval-Augmentedd Generation can be used in educational applications.
The main objectives were to:
- Build a complete RAG pipeline
- Perform semantic document retrieval
- Work with vecctor embeddings
- Use FAISS for similarity search
- Ground AI responses in external documents
- Build AI generated learning tools
- Develop an interactive Streamlit application
- Combine NLp, information retrieval, and generative AI in one system

## Limitations
- Scanned PDFs without readable text may require OCR.
- Processing very large documents can take additional time.
- AI-generated study content can still contain inaccuracies.
- Retrieval quality depends on the document content and query.
- Semantic similarity does not guarantee that every retrieved section is perfectly relevant.

## Possible Future improvements

Potential extensions include:
- Support for additional document formats
- Persistent vector databases
- Conversation memory
- Difficulty-controlled quizzes
- Exportable flashcards
- Multi-document citation improvements
- Faster document indexing
- Additional embedding models
These features were intentionally kept outside the current scope to maintain a focused study-assistant workflow.

## Project Status
Core functionlity implemented:
- PDF upload
- Text extraction
- Text chunking
- Embedding generation
- FAISS vector indexing
- Semantic retrieval
- Grounded question answering
- SOurce-page reference
- Topic summarization
- Quiz generation
- QUiz scoring and explanations
- Flashcard generation
- Interactive Streamlit interface
- Custom UI styling






Notes
# Text chunks in RAG
when i upload the pdf file, texxt is divided into chunks according to the chunk_size and chunk_overlap settings in rag_pipeline.py.
but careful in dividing, 
chunl size directly affects answer quality.
- if chunks are too small, important context get separated.
- if chunks are too large, irrelevant information enters the answer.
- overlap helps preserve meaning between neighboring chunks.

# Chunk vs vector
chunk is the actual words that human can read. ( how is LIFO? )
vector is the numbers representing for computers can read. ( 0, 1, 2.2 )

# Hugging face
- online platform for AI models
Developers can use it to:
- find pretrained AI models
- Download and fine-tune models
- Run models through an API
- share datasets and AI applications
In our projects,
- Sentence Transformer model: convert text into chunk
- Hugging face : will run an LLM that turns the chunks into clear answsers.
We need a free Hugging face account to obtain a private access token for the LLM.
Never push that token to github.

# 

