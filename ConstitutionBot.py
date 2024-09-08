from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader, StorageContext, load_index_from_storage
import os
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from langchain.prompts import ChatPromptTemplate
from langchain.memory.buffer import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

class ConstitutionBot:
    def __init__(self, model_name):
        load_dotenv()
        self.memory = ConversationBufferMemory()
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=model_name
        )
    
    def rewrite_query(self, query):
        rewriting_prompt_str = """
        [INST] 
        Given the Chat History and Question, rephrase the question so it can be a standalone question.  I will give some examples.

        ```
        Example 1: If no chat history is present, then just return the original question
        Chat History: 
        Question: What is the power of the prime minister?
        Your response: What is the power of the prime minister?
        ```

        ```
        Example 2: 
        Chat History: 
        Human: What is the power of the prime minister?
        AI: The prime minister has many powers
        Question: What else can he do?
        Your response: What else can the prime minister do? 
        ```
        Example 3: 
        Chat History: 
        Human: Who appoints the prime minister in a no contest?
        AI: The govenor general
        Question: What is their role?
        Your response: What is the govenor general's role? 
        ```

        With those examples, here are the actual chat history and input questions:
        Chat History: {chat_history}
        Question: {question}
        [/INST] 
        """
        rewriting_prompt = ChatPromptTemplate.from_template(rewriting_prompt_str)
        rewritten_question = self.llm.invoke(rewriting_prompt.invoke({"chat_history":self.memory.chat_memory.__str__(), "question":query}).to_string()).content
        return rewritten_question

    def retrieve_documents(self, query):
        if os.path.isdir("index"):
            storage_context = StorageContext.from_defaults(persist_dir="index")
            index = load_index_from_storage(storage_context)
        else:
            documents=SimpleDirectoryReader("data").load_data()
            index=VectorStoreIndex.from_documents(documents,show_progress=True)
            index.storage_context.persist(persist_dir="index")
        
        retriever = index.as_retriever(similarity_top_k=10)
        nodes = retriever.retrieve(query)

        reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5)
        reranked_nodes = reranker.postprocess_nodes(nodes=nodes, query_str=query)
        combined_context = "\n".join(node.text for node in reranked_nodes)
        return combined_context
            

    def query(self, query):
        standalone_query = self.rewrite_query(query)
        context = self.retrieve_documents(standalone_query)
        

        final_prompt_str = """
        [INST] 
        Given the following chat history and context, answer the following question.

        Context: {context}
        Chat History: {chat_history}
        
        {question}
        [/INST] 
        """

        final_prompt = ChatPromptTemplate.from_template(final_prompt_str)
        response = self.llm.invoke(final_prompt.invoke({"context":context, "chat_history":self.memory.chat_memory.__str__(), "question":standalone_query}).to_string()).content
        print(f"Constitution Bot: {response}")
        self.memory.save_context({"HumanMessage":standalone_query},{"AIMessage":response})

        #print(f"memory: {self.memory.chat_memory.__str__()}")

if __name__ == "__main__":
    bot = ConstitutionBot("gpt-3.5-turbo")
    print("Hi! I'm Constitution Bot! I know everything about Australia's Consitution. Ask me a question :) \nType 'exit' to end the conversation")
    while True:
        query = input("Enter your question:")
        if query.lower() == 'exit':
            break
        bot.query(query)
