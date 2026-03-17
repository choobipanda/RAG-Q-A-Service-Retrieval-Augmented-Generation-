7. Why does grounding reduce hallucinations?
Grounding reduces hallucinations because the LLM is given retrieved document context to base its answer on, instead of relying only on its general knowledge. This helps the model produce responses that are supported by the ingested content rather than making up information.
8. How do chunk size and overlap affect retrieval quality?
Chunk size affects how much information is stored in each section of text. Smaller chunks can make retrieval more precise, while larger chunks can preserve more context. Overlap helps keep important information from being split between chunks, which improves the chances of retrieving complete and relevant context.
9. What is the difference between semantic search and keyword search?
Semantic search finds results based on meaning, even if the exact words in the query do not appear in the text. Keyword search looks for exact word matches, so it may miss relevant content if different wording is used.
10. What are common failure modes of RAG systems?
Common failure modes of RAG systems include retrieving irrelevant chunks, missing the most useful chunk, using chunks that do not contain enough context, and generating answers that go beyond the retrieved evidence. Poor chunking, weak embeddings, or unclear prompts can all reduce answer quality.
11. Why are citations important in AI systems?
Citations are important because they show which retrieved chunks were used to support the answer. This makes the response more transparent, helps users verify the information, and increases trust in the system.