# TextSummarizationLocalLLM

Refer to the Local LLM_IA_IMH.pdf for the project introduction

## 1)llama-cpp-llm - LangChainV1.ipynb

  -This code is the basics for install and testing the local LLM.

  -Run the code in the jupyter lab notebook

## 2)app_llm_sum.py
 
 -This code run on streamlit UI to test the text summarization with the local LLM

 -There will be a chatbot interface with system prompt

 -You can edit the system prompt for your own purpose


## 3)app_llm_sumv2.py

 -This code run on streamlit UI to do text summarization by input a PDF file

 -Current version only support PDF file content with only text

 -There are 3 summarization techniques use in the code mapreduce, mapreduce(with text chucking) and refine(cumulative summary of smaller chuck)

## 4)app_rag.py

-This code run on streamlit UI to do question and answer on the loaded PDF file

-Current version only support PDF file content with only text

-It used the embedding model to convert text into Chroma vector for searching of information based on the user prompt question

-Download the embedding model https://drive.google.com/file/d/1j_C1CXpvHHmj42qte4TOsq_iAztI-TLF/view?usp=sharing
Save in /models/embedding


## Evaluate the LLM Summarization task

 - [examples/evaluation/How_to_eval_abstractive_summarization.ipynb](https://cookbook.openai.com/examples/evaluation/how_to_eval_abstractive_summarization)


