import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, Dict, Any
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
import nest_asyncio
import logging
import time
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indian_tax_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IndianTaxAgent')

# Enable nested asyncio and load environment variables
nest_asyncio.apply()
load_dotenv()

@dataclass
class Config:
    qdrant_url: str = os.getenv("QDRANT_URL_LOCALHOST", "http://localhost:6333")
    serper_api_key: str = os.getenv("SERPER_API_KEY")
    model_name: str = "gemini-pro"
    analysis_model_name: str = "deepseek-r1:latest"
    collection_name: str = "ITR-4"

class ProgressManager:
    def __init__(self, total_steps: int):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.total_steps = total_steps
        self.current_step = 0

    def update(self, message: str):
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        self.status_text.text(f"Step {self.current_step}/{self.total_steps}: {message}")
        logger.info(f"Progress: {message} ({self.current_step}/{self.total_steps})")

    def complete(self):
        self.progress_bar.progress(1.0)
        self.status_text.text("Processing complete!")
        time.sleep(1)
        self.progress_bar.empty()
        self.status_text.empty()

def parse_response(response: str, deep_search: bool = False) -> str:
    """
    Clean and format the response text.
    For deep research (when deep_search is True), remove any <think>...</think> tags.
    """
    if deep_search:
        # Remove content between <think> and </think> tags (non-greedy, across newlines)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    try:
        # Try to parse as JSON first (in case it still returns JSON)
        data = json.loads(response)
        if isinstance(data, dict):
            output_parts = []
            if "direct_answer" in data:
                output_parts.append(data["direct_answer"])
            if "detailed_explanation" in data:
                output_parts.append(data["detailed_explanation"])
            if "additional_notes" in data:
                output_parts.append(f"Note: {data['additional_notes']}")
            return "\n\n".join(output_parts)
    except json.JSONDecodeError:
        # Clean up the text response
        cleaned = response.replace("```", "").replace("{", "").replace("}", "")
        cleaned = re.sub(r'"[^"]*":\s*', "", cleaned)  # Remove JSON-like keys
        cleaned = re.sub(r'[\[\]]', "", cleaned)  # Remove brackets
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Remove extra newlines
        return cleaned.strip()

def format_display_result(result: str) -> str:
    """
    Format the result for better display in Streamlit by properly handling Markdown formatting.
    """
    # Clean up any double/triple asterisks that aren't properly spaced
    result = re.sub(r'\*{2,3}([^*]+)\*{2,3}', r'**\1**', result)
    
    sections = result.split('\n\n')
    formatted_sections = []
    
    for section in sections:
        if section.strip():
            # Handle section headers: if a colon exists, treat text before as header
            if ':' in section:
                title, content = section.split(':', 1)
                title = title.replace('*', '')
                formatted_sections.append(f"### {title.strip()}\n{content.strip()}")
            else:
                formatted_sections.append(section)
    
    formatted_result = "\n\n".join(formatted_sections)
    
    # Additional cleanup for any remaining improper markdown
    formatted_result = re.sub(r'\*{4,}', '**', formatted_result)
    formatted_result = re.sub(r'\*{3}', '**', formatted_result)
    formatted_result = re.sub(r'\*\*\s+\*\*', '', formatted_result)
    
    return formatted_result

class IndianTaxAgent:
    def __init__(self, config: Config):
        logger.info("Initializing IndianTaxAgent")
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url)
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.analysis_llm = OllamaLLM(model=config.analysis_model_name)
        self.embeddings = FastEmbedEmbeddings()
        self.setup_tools_and_agents()
        logger.info("IndianTaxAgent initialization complete")

    def setup_tools_and_agents(self):
        try:
            logger.info("Setting up tools and agents")
            self.search_tool = Tool(
                name="IndianTaxResearch",
                func=GoogleSerperAPIWrapper(serper_api_key=self.config.serper_api_key).run,
                description="Searches for latest Indian tax regulations and ITR guidelines"
            )
            self.analysis_tool = Tool(
                name="ITRAnalysis",
                func=self.retrieve_itr,
                description="Analyzes Indian Income Tax Return (ITR) data and regulations"
            )
            self.research_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.analysis_memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.research_agent = initialize_agent(
                tools=[self.search_tool],
                llm=self.llm,
                agent="zero-shot-react-description",
                memory=self.research_memory,
                handle_parsing_errors=True,
                verbose=True
            )
            # Set handle_parsing_errors=True for the analysis agent as well.
            self.analysis_agent = initialize_agent(
                tools=[self.analysis_tool],
                llm=self.analysis_llm,
                agent="zero-shot-react-description",
                memory=self.analysis_memory,
                handle_parsing_errors=True,
                verbose=True
            )
            logger.info("Tools and agents setup complete")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise

    def retrieve_itr(self, query: str) -> str:
        logger.info(f"Retrieving ITR data for query: {query}")
        try:
            vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.config.collection_name,
                url=self.config.qdrant_url
            )
            docs = vectorstore.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(query)
            if not docs:
                logger.warning("No relevant ITR data found")
                return "No relevant Indian ITR data found."
            logger.info(f"Retrieved {len(docs)} documents")
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error retrieving ITR data: {str(e)}")
            return f"Error retrieving ITR data: {str(e)}"

    def enhance_output_with_llm(self, raw_result: str, query: str) -> str:
        logger.info("Enhancing output with LLM")
        try:
            prompt = f"""
You are an expert Indian Tax Assistant specializing in Indian Income Tax Returns (ITR) and you have been provided with indian tax related data before. 
Analyze the following information and provide a detailed response focused on Indian tax laws and ITR guidelines.

Important guidelines:
1. Only provide information relevant to Indian taxation system
2. Reference specific ITR sections and forms where applicable
3. Include recent Indian tax updates if relevant
4. All amounts should be in INR (₹)
5. Cite specific Indian Income Tax Act sections when relevant

Query: {query}

Raw Information:
{raw_result}

Your response should follow this structure:
1. Main Answer: Clear, direct answer to the query
2. Applicable ITR Details: Relevant forms, sections, and deadlines
3. Calculation Method (if applicable): Step-by-step breakdown for any calculations
4. Additional Notes: Important considerations specific to Indian taxation

Use proper markdown formatting with headers (###) and bold (**) where appropriate.
"""
            enhanced_response = self.llm.predict(prompt)
            logger.info("LLM enhancement completed")
            return enhanced_response
        except Exception as e:
            logger.error(f"Error enhancing output: {str(e)}")
            return f"Error processing query: {str(e)}"

    def process_quick_search(self, query: str) -> str:
        logger.info(f"Starting quick search for query: {query}")
        progress = ProgressManager(total_steps=3)
        try:
            progress.update("Retrieving relevant ITR documents")
            raw_result = self.retrieve_itr(query)
            progress.update("Analyzing ITR information")
            enhanced_response = self.enhance_output_with_llm(raw_result, query)
            progress.update("Formatting response")
            progress.complete()
            
            cleaned_response = parse_response(enhanced_response, deep_search=False)
            return cleaned_response
        except Exception as e:
            logger.error(f"Error in quick search: {str(e)}")
            progress.complete()
            return f"Error in quick search: {str(e)}"

    async def process_deep_research_async(self, query: str) -> str:
        logger.info(f"Starting deep research for query: {query}")
        progress = ProgressManager(total_steps=4)
        try:
            progress.update("Initiating tax research analysis")
            research_task = self._run_agent(self.research_agent, query)
            progress.update("Analyzing ITR regulations")
            analysis_task = self._run_agent(self.analysis_agent, query)
            progress.update("Processing results")
            research_result, analysis_result = await asyncio.gather(research_task, analysis_task)
            
            combined_result = f"""
Indian Tax Research Findings:
{research_result}

ITR Analysis:
{analysis_result}
"""
            
            progress.update("Preparing final response")
            enhanced_response = self.enhance_output_with_llm(combined_result, query)
            progress.complete()
            
            # For deep research, remove any <think></think> sections.
            cleaned_response = parse_response(enhanced_response, deep_search=True)
            return cleaned_response
        except Exception as e:
            logger.error(f"Error in deep research: {str(e)}")
            progress.complete()
            return f"Error in deep research: {str(e)}"

    def process_deep_research(self, query: str) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.process_deep_research_async(query))

    async def _run_agent(self, agent, query: str) -> str:
        # Offload the potentially blocking synchronous call to a separate thread.
        return await asyncio.to_thread(agent.run, query)

def initialize_session_state():
    if "tax_agent" not in st.session_state:
        logger.info("Initializing new IndianTaxAgent in session state")
        config = Config()
        st.session_state.tax_agent = IndianTaxAgent(config)
    if "history" not in st.session_state:
        st.session_state.history = []

def main():
    logger.info("Starting Indian Tax Return Assistant application")
    st.set_page_config(
        page_title="Indian Tax Return Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    initialize_session_state()
    
    st.title("Indian Tax Return Assistant")
    st.markdown("""
    <style>
        .stTitle {
            color: #FF9933;
            text-align: center;
        }
        .tax-result {
            background-color: #000000;
            border-left: 4px solid #FF9933;
            padding: 20px;
            margin: 10px 0;
            border-radius: 5px;
            color: #fff;
        }
        .tax-result h3 {
            color: #138808;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .tax-result strong {
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Search Settings")
        search_mode = st.radio("Select Search Mode", ("Quick Search", "Deep Research"))
        st.markdown("### Recent Logs")
        try:
            with open('indian_tax_agent.log', 'r') as log_file:
                recent_logs = log_file.readlines()[-5:]
                for log in recent_logs:
                    st.text(log.strip())
        except Exception:
            st.text("No logs available.")

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "Enter your tax-related query:",
            placeholder="e.g., What are the latest deductions under Section 80C?"
        )
        if st.button("Search", type="primary"):
            if not query:
                st.warning("Please enter a query first.")
                return
            tax_agent = st.session_state.tax_agent
            try:
                if search_mode == "Deep Research":
                    result = tax_agent.process_deep_research(query)
                else:
                    result = tax_agent.process_quick_search(query)
                
                formatted_result = format_display_result(result)
                
                st.session_state.history.append({
                    "query": query,
                    "result": formatted_result,
                    "mode": search_mode,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.success("Search completed!")
                st.markdown("### Results")
                
                # Display the final formatted output directly as Markdown
                st.markdown(formatted_result, unsafe_allow_html=True)
                
                # Optionally, display the raw retrieved documents in an expander.
                with st.expander("Retrieved Documents from Qdrant"):
                    raw_itr_data = tax_agent.retrieve_itr(query)
                    st.text(raw_itr_data)
                    
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

    with col2:
        st.markdown("### Search History")
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"{item['timestamp']} - {item['query'][:50]}..."):
                st.markdown(f"*Mode:* {item['mode']}")
                st.markdown(item['result'], unsafe_allow_html=True)

if __name__ == "__main__":
    main()
