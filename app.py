# Standard Library Imports
import logging
import os

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import gradio as gr
from huggingface_hub import snapshot_download

# LlamaIndex (Formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class CrossEncoderRerankRetriever:
    def __init__(self, base_retriever, cross_encoder, top_k=10):
        self.base_retriever = base_retriever
        self.cross_encoder = cross_encoder
        self.top_k = top_k

    async def aretrieve(self, query: str):
        # Step 1: get candidate documents
        docs = await self.base_retriever.aretrieve(query)
        texts = [doc.text for doc in docs]

        # Step 2: prepare query-document pairs
        pairs = [[query, doc] for doc in texts]

        # Step 3: rerank using CrossEncoder
        scores = self.cross_encoder.predict(pairs)

        # Step 4: sort and select top_k
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked_docs = [docs[i] for i in ranked_indices[:self.top_k]]

        return reranked_docs


load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPT_SYSTEM_MESSAGE = """You are a financial news analyst AI, helping users understand and stay updated on the latest developments in financial markets.
Your focus is on interpreting and summarizing live financial news, market movements, macroeconomic updates, corporate earnings, and key financial indicators.
Topics may include stocks, bonds, commodities, foreign exchange, central bank announcements, inflation data, and geopolitical events influencing markets.

To ensure accuracy and relevance, always rely on the "Live_Financial_News_Sources" tool to gather information for your responses.

Not all content retrieved by the tool may be relevant to a given question — carefully filter and use only the information that directly addresses the user's query.
Do not supplement your answers with knowledge not contained in the tool’s responses.

If the user requests a deeper analysis or more context around a previously discussed topic, you should reformulate your input to the tool to reflect the new focus or level of detail.
Provide clear, structured, and informative answers that help users understand both the news content and its broader financial implications.

If the tool does not return information relevant to the question, politely explain that the current news data does not cover the topic, and therefore you cannot provide an answer.

Present your responses in Markdown format for readability, and always encourage users to ask follow-up questions if they seek more detailed insights into any aspect of the financial news."""


TEXT_QA_TEMPLATE = """
You must answer only questions related to financial news, market updates, and economic developments.
Always leverage the retrieved live financial news sources to answer the questions — do not answer them based on your own knowledge or assumptions.
If the query is not relevant to financial news, market trends, or economic reports, say that you don't know the answer.
"""


class FinancialLLM:

    def __init__(self):
        self.tools = None
        Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")



    def get_tools(self, data_folder: str, db_collection: str):
        # Step 1: Load documents from .txt files
        documents = SimpleDirectoryReader(data_folder).load_data()

        # Step 2: Setup Chroma persistence if you want to reuse it later
        db = chromadb.PersistentClient(path=f"data/{db_collection}")
        chroma_collection = db.get_or_create_collection(db_collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Step 3: Build the index
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            show_progress=True,
            use_async=True,
            embed_model=Settings.embed_model,
        )

        # Step 4: Setup retriever
        # Original retriever
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=30,  # Fetch more so reranker can filter better
            embed_model=Settings.embed_model,
            use_async=True,
        )

        # Create CrossEncoder reranker
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        def rerank_nodes(query: str, nodes):
            """Rerank nodes using CrossEncoder."""
            if not nodes:
                return nodes

            # Prepare sentence pairs for cross-encoder
            pairs = [(query, node.get_content()) for node in nodes]

            # Get scores from cross-encoder
            scores = cross_encoder.predict(pairs)

            # Sort nodes based on cross-encoder scores
            reranked_nodes = [
                node for _, node in sorted(
                    zip(scores, nodes),
                    key=lambda x: x[0],
                    reverse=True
                )
            ]

            return reranked_nodes[:15]  # Return top 15 after reranking

        # Create a retriever that applies reranking
        class RerankerRetriever(BaseRetriever):
            def __init__(self, base_retriever, rerank_fn):
                self.base_retriever = base_retriever
                self.rerank_fn = rerank_fn

            def _retrieve(self, query_bundle):
                nodes = self.base_retriever.retrieve(query_bundle)
                a = self.rerank_fn(query_bundle.query_str, nodes)
                return a

        # Create the final retriever
        final_retriever = RerankerRetriever(
            base_retriever=vector_retriever,
            rerank_fn=rerank_nodes
        )

        # Step 5: Wrap it as a tool
        tools = [
            RetrieverTool(
                retriever=final_retriever,
                metadata=ToolMetadata(
                    name="Text_Files_Information",
                    description="Useful for retrieving information from the loaded .txt documents.",
                ),
            )
        ]
        self.tools = tools




    def generate_completion(self, query, history, memory):
        logging.info(f"User query: {query}")

        # Manage memory
        chat_list = memory.get()
        if len(chat_list) != 0:
            user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
            if len(user_index) > len(history):
                user_index_to_remove = user_index[len(history)]
                chat_list = chat_list[:user_index_to_remove]
                memory.set(chat_list)
        logging.info(f"chat_history: {len(memory.get())} {memory.get()}")
        logging.info(f"gradio_history: {len(history)} {history}")

        if self.tools is None:
            raise ValueError("Please define tools")
        agent = OpenAIAgent.from_tools(
            llm=Settings.llm,
            memory=memory,
            tools=self.tools,
            system_prompt=PROMPT_SYSTEM_MESSAGE,
        )

        # Generate answer
        completion = agent.stream_chat(query)
        answer_str = ""
        for token in completion.response_gen:
            answer_str += token
            yield answer_str


    def launch_ui(self):
        with gr.Blocks(
            fill_height=True,
            title="AI Tutor 🤖",
            analytics_enabled=True,
        ) as demo:

            memory_state = gr.State(
                lambda: ChatSummaryMemoryBuffer.from_defaults(
                    token_limit=120000,
                )
            )
            chatbot = gr.Chatbot(
                scale=1,
                placeholder="<strong>AI Tutor 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
                show_label=False,
                show_copy_button=True,
            )

            gr.ChatInterface(
                fn=self.generate_completion,
                chatbot=chatbot,
                additional_inputs=[memory_state],
            )

            demo.queue(default_concurrency_limit=64)
            demo.launch(debug=True, share=False) # Set share=True to share the app online

if __name__ == "__main__":
    finance_llm = FinancialLLM()
    tools = finance_llm.get_tools(db_collection="scraped_news", data_folder="data/scraped_news")
    finance_llm.launch_ui()
