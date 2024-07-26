# streamlit run app_llm_sumv2.py

#Directory structure to store the LLM mode gguf
# models/llm/
# amethyst-13b-mistral.Q6_K
# mistral-7b-instruct-v0.1.Q6_K
# gemma-7b-it.Q6_K
# orca-2-7b.Q6_K


from typing import List, Union, Optional

# from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from PyPDF2 import PdfReader
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import TokenTextSplitter
from langchain.chains import MapReduceDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import WebBaseLoader
from langchain.chains import LLMChain

from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from datetime import datetime

PROMPT_TEMPLATE = """
Use the following pieces of context enclosed by triple backquotes to answer the question at the end.
\n\n
Context:
```
{context}
```
\n\n
Question: [][][][]{question}[][][][]
\n
Answer:"""

doc_type="pdf"


def init_page() -> None:
    st.set_page_config(
        page_title="Personal ChatGPT"
    )
    st.sidebar.title("Options")
    # doc_type =  st.sidebar.radio("Choose data type:",
    #                               ("pdf",
    #                                "txt",))
    


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content=(
                    "You are a helpful AI QA assistant. "
                    "When answering questions, use the context enclosed by triple backquotes if it is relevant. "
                    "If you don't know the answer, just say that you don't know, "
                    "don't try to make up an answer. "
                    "Reply your answer in mardkown format.")
            )
        ]
        st.session_state.costs = []


#get_pdf_text()
#get the pdf file and performance the text summarization with selected summarization technique

def get_pdf_text(llm) -> Optional[str]:
    """
    Function to load PDF text and split it into chunks.
    """
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        label="Here, upload your file you want ChatGPT to use to answer",
        type= None
    )
    if uploaded_file:
        print(uploaded_file)
        print(uploaded_file.name)

        if doc_type=="pdf":
            #pdf file loader
            # pdf_reader = PdfReader(uploaded_file)
            # text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
            
            loader = PyPDFLoader(uploaded_file.name)
            pages = loader.load_and_split()
            text = pages
            
        else:
            #Text file loader
            loader = TextLoader(uploaded_file.name)       
            text = loader.load()
       

        #3 type of summarizarion are tested here
        # do_summarization() - use map reduce method, but customised with text split method
        # do_summarization_mapreduce -  use longchain text summarization mapreduce
        # do_summarizattion_refine - use langcahain acculative text summarizatin 

        start_time = datetime.now()
        
        summary_text= do_summarization(text,llm)

        # summary_text=do_summarization_refine(text,llm)

        # summary_text=do_summarization_mapreduce(text,llm)

        time_diff= (datetime.now()  - start_time).total_seconds()

        st.session_state.costs.append(str(time_diff))
        print("Execution time of program is: ", time_diff, "s")
        
        return None 

    else:
        return None

def do_summarization( text_input, llm):
    # 1. Map Chain
    map_template = """
    Write a concise summary of the following content:
    
    {content}
    
    Summary:
    """

    map_prompt = PromptTemplate.from_template(map_template)
    
    # 2. Reduce Chain
    # reduce_template = """The following is set of summaries:
    
    # {doc_summaries}
    
    # Summarize the above summaries with into more concise.
    # Ensure that the summaries have no repeated information. 
    # Summary:
    
    # """
    
    # reduce_template = """The following is set of summaries:
    # {doc_summaries}
    # Take these and distill it into a final, consolidated summary with informative information. 
    # Summary:"""

    reduce_template = """The following is set of summaries:
    {doc_summaries}

    Extract and organize the data in a structured format, ensuring that the output is accurate, comprehensive, and suitable for further analysis or integration with other healthcare systems.
    Take these and distill it into a final, consolidated  informative summary report. Ensure the report is complete without partial information.
    Report:
    """

    
    map_chain = LLMChain(prompt=map_prompt, llm=llm)

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(prompt=reduce_prompt, llm=llm)
    
    
    stuff_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries")
    
    reduce_chain = ReduceDocumentsChain(
        combine_documents_chain=stuff_chain,
    )

    # 3. Map Reduce Chain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        document_variable_name="content",
        reduce_documents_chain=reduce_chain
    )

    # Split the content into smaller chuncks
    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(text_input)
    # print("no of split doc = ", len(split_docs))

    # summary = map_reduce_chain.run(split_docs)
    pages=text_input
    summary = map_reduce_chain({"input_documents": pages})
    print(summary['output_text'])
    with st.spinner("ChatGPT is typing ..."):
        st.session_state.messages.append(AIMessage(content=summary['output_text']))
    return summary


def do_summarization_refine( text_input, llm):
    question_prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["text"]
    )
    
    refine_prompt_template = """
                  Write a concise summary of the following text delimited by triple backquotes.
                  Return your response in paragraphs which covers the the important information.
                  ```{text}```
                  SUMMARY:
                  """
    
    refine_prompt = PromptTemplate(
        template=refine_prompt_template, input_variables=["text"]
    )

    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    )

    pages=text_input
    summary = refine_chain({"input_documents": pages})
    with st.spinner("ChatGPT is typing ..."):
        st.session_state.messages.append(AIMessage(content=summary['output_text']))
    return summary



def do_summarization_mapreduce( text_input, llm):
    map_prompt_template = """
                      Write a summary of this chunk of text that includes the main points and any important details.
                      {text}
                      """

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    
    combine_prompt_template = """
                          Write a concise summary of the following text delimited by triple backquotes.
                         Return your response in paragraphs which covers the the important information.
                         ```{text}```
                          SUMMARY:
                          """
    combine_prompt = PromptTemplate(
    template=combine_prompt_template, input_variables=["text"]
)
    map_reduce_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    return_intermediate_steps=True,
    )

    pages=text_input
    summary = map_reduce_chain({"input_documents": pages})

    with st.spinner("ChatGPT is typing ..."):
            st.session_state.messages.append(AIMessage(content=summary['output_text']))
    return summary


    

def build_vectore_store(
    texts: str, embeddings: Union[OpenAIEmbeddings, LlamaCppEmbeddings]) \
        -> Optional[Qdrant]:
    """
    Store the embedding vectors of text chunks into vector store (Qdrant).
    """
    if texts:
        with st.spinner("Loading PDF ..."):
            qdrant = Qdrant.from_texts(
                texts,
                embeddings,
                path=":memory:",
                collection_name="my_collection",
                force_recreate=True
            )
        st.success("File Loaded Successfully!!")
    else:
        qdrant = None
    return qdrant


def select_llm() -> Union[ChatOpenAI, LlamaCpp]:
    """
    Read user selection of parameters in Streamlit sidebar.
    """
   
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("mistral-7b-instruct-v0.1.Q6_K",
                                   "gemma-7b-it.Q6_K",
                                   "orca-2-7b.Q6_K",
                                   "amethyst-13b-mistral.Q6_K",))
    
    # doc_type =  st.sidebar.radio("Choose data type:",
    #                               ("pdf",
    #                                "txt",))

    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)

    
    return model_name,temperature


def load_llm(model_name: str, temperature: float) -> Union[ChatOpenAI, LlamaCpp]:
    """
    Load LLM.
    """
    if model_name.startswith("gpt-"):
        return ChatOpenAI(temperature=temperature, model_name=model_name)
    else:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            model_path=f"./models/llm/{model_name}.gguf",
            input={"temperature": temperature,
                   "max_length": 8000,
                   "top_p": 1
                   },
            max_tokens = 8000,
            n_ctx=8000,
            n_gpu_layers=-1,
            n_batch=512,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=False,  # True
            streaming=True,
        )


def load_embeddings(model_name: str) -> Union[OpenAIEmbeddings, LlamaCppEmbeddings]:
    """
    Load embedding model.
    """

    if model_name.startswith("gpt-"):
        return OpenAIEmbeddings()
    else:
        embed_model_id = ".\\models\\embedding\\all-MiniLM-L6-v2"
        embed_model = HuggingFaceEmbeddings(model_name=embed_model_id)
        # return LlamaCppEmbeddings(model_path=f".//models//llm//mistral-7b-instruct-v0.1.Q6_K.gguf")
        return embed_model


def get_answer(llm, messages) -> tuple[str, float]:
    """
    Get the AI answer to user questions.
    """
    if isinstance(llm, ChatOpenAI):
        with get_openai_callback() as cb:
            answer = llm(messages)
        return answer.content, cb.total_cost
    if isinstance(llm, LlamaCpp):
        return llm(llama_v2_prompt(convert_langchainschema_to_dict(messages))), 0.0
        # return llm.invoke(messages), 0.0


def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant
    format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)


def extract_userquesion_part_only(content):
    """
    Function to extract only the user question part from the entire question
    content combining user question and pdf context.
    """
    content_split = content.split("[][][][]")
    if len(content_split) == 3:
        return content_split[1]
    return content


def main() -> None:
    # _ = load_dotenv(find_dotenv())

    init_page()

    model_name, temperature = select_llm()
    llm = load_llm(model_name, temperature)
    # embeddings = load_embeddings(model_name)

    texts = get_pdf_text( llm )
    # qdrant = build_vectore_store(texts, embeddings)

    init_messages()

    st.header("LLMChat Summarizer")
    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        if qdrant:
            context = [c.page_content for c in qdrant.similarity_search(
                user_input, k=10)]
            user_input_w_context = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]) \
                .format(
                    context=context, question=user_input)
        else:
            user_input_w_context = user_input
        st.session_state.messages.append(
            HumanMessage(content=user_input_w_context))
        with st.spinner("ChatGPT is typing ..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        # st.session_state.costs.append(cost)

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(extract_userquesion_part_only(message.content))

    # costs = st.session_state.get("costs", [])
    # st.sidebar.markdown("## Costs")
    # st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    # for cost in costs:
    #     st.sidebar.markdown(f"- ${cost:.5f}")


# streamlit run app.py
if __name__ == "__main__":
    main()