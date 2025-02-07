# Chainlit

## Setup
1. Clone the repo and navigate to the folder
    ```
    git clone https://github.com/ZXEcoder/New_ITR_Filing_AI-assistant.git
    
    ```

2. Make sure you have `uv` installed,
    - If not installed, here is the website [link](https://docs.astral.sh/uv/getting-started/installation/)
      

3. Install the necessary packages, it also creats the virtual environment for you.
    ```
    uv sync
    ```

4. We will use Qdrant locally, so lets provide qdrant url in the `.env` file.

    TO RUN QDRANT LOCALLY, INSTALL [DOCKER](https://www.docker.com/get-started/) FIRST, HERE IS THE [LINK](https://qdrant.tech/documentation/quickstart/)
    ```
    docker pull qdrant/qdrant
    docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
    
    ```
    - Rename `.env.example` to `.env`
    - Provide your env variables inside it as shown below.
    ```
    QDRANT_URL_LOCALHOST="xxxxx"
    ```
  

4. Run the chainlit app
    ```
    uv run ingest.py
    uv run chainlit run rag-chainlit-deepseek.py
    ```

    
5. For more info follow along with me in the video.

---
Important Links:
- https://ds4sd.github.io/docling/examples/rag_langchain/
- https://blog.gopenai.com/how-to-build-a-chatbot-to-chat-with-your-pdf-9abb9beaf0c4

 Happy learning 😎
