"# 19_AI_study_partner

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

