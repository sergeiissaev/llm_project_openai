
import logging

from dotenv import load_dotenv
import chromadb
import gradio as gr

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
        self.valid_key = False
        Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")



    def get_tools(self, data_folder: str, db_collection: str):
        documents = SimpleDirectoryReader(data_folder).load_data()

        db = chromadb.PersistentClient(path=f"data/{db_collection}")
        chroma_collection = db.get_or_create_collection(db_collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            show_progress=True,
            use_async=True,
            embed_model=Settings.embed_model,
        )

        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=30,
            embed_model=Settings.embed_model,
            use_async=True,
        )

        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        def rerank_nodes(query: str, nodes):
            """Rerank nodes using CrossEncoder."""
            if not nodes:
                return nodes

            pairs = [(query, node.get_content()) for node in nodes]

            scores = cross_encoder.predict(pairs)

            reranked_nodes = [
                node for _, node in sorted(
                    zip(scores, nodes),
                    key=lambda x: x[0],
                    reverse=True
                )
            ]

            return reranked_nodes[:15]

        class RerankerRetriever(BaseRetriever):
            def __init__(self, base_retriever, rerank_fn):
                self.base_retriever = base_retriever
                self.rerank_fn = rerank_fn

            def _retrieve(self, query_bundle):
                nodes = self.base_retriever.retrieve(query_bundle)
                a = self.rerank_fn(query_bundle.query_str, nodes)
                return a

        final_retriever = RerankerRetriever(
            base_retriever=vector_retriever,
            rerank_fn=rerank_nodes
        )

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
        if self.valid_key:
            logging.info(f"User query: {query}")

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

            completion = agent.stream_chat(query)
            answer_str = ""
            for token in completion.response_gen:
                answer_str += token
                yield answer_str
        else:
            yield "Please enter an API key"


    def set_openai_key(self, api_key):
        """Update the OpenAI settings with user-provided API key"""
        self.openai_key = api_key
        if api_key:
            try:
                Settings.llm = OpenAI(
                    api_key=api_key,
                    temperature=0,
                    model="gpt-4"
                )
                Settings.embed_model = OpenAIEmbedding(
                    api_key=api_key,
                    model="text-embedding-3-small"
                )
                return True, "API key set successfully"
            except Exception as e:
                return False, f"Error setting API key: {str(e)}"
        return False, "No API key provided"

    def launch_ui(self):
        with gr.Blocks(
            fill_height=True,
            title="AI Tutor 🤖",
            analytics_enabled=True,
        ) as demo:
            with gr.Row():
                openai_key = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    info="Required for all OpenAI functionality",
                    scale=4
                )
                api_status = gr.HTML("<div style='color: gray'>⚪ No API key provided</div>")

                def verify_and_set_key(key):
                    success, message = self.set_openai_key(key)
                    status_icon = "🟢" if success else "🔴" if key else "⚪"
                    color = "green" if success else "red" if key else "gray"
                    if success:
                        self.valid_key = True
                        self.get_tools(db_collection="scraped_news", data_folder="data/scraped_news")
                    return f"<div style='color: {color}'>{status_icon} {message}</div>"

                openai_key.change(
                    fn=verify_and_set_key,
                    inputs=openai_key,
                    outputs=api_status
            )


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
        demo.launch(debug=True, share=False)

if __name__ == "__main__":
    finance_llm = FinancialLLM()
    #tools = finance_llm.get_tools(db_collection="scraped_news", data_folder="data/scraped_news")
    finance_llm.launch_ui()
