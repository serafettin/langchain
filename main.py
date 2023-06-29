from flask import Flask, make_response, jsonify
import os
from pprint import pprint
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredURLLoader
app = Flask(__name__)


@app.route("/indexes/<index_id>/documents/<path:url>", methods=["POST"])
def add_document(index_id, url):
    #payload = request.json
    #url = payload['url']
    local_directory = "kb-h2o-wave-3"
    persist_directory = os.path.join(os.getcwd(), local_directory)
    loader = UnstructuredURLLoader(urls=[url])
    kb_data = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    kb_doc = text_splitter.split_documents(kb_data)
    embeddings = OpenAIEmbeddings()
    ##If exists return
    ids = [f'{i}-{url}' for i in range(1, len(kb_doc) + 1)]
    print(ids)
    kb_db1 = Chroma.from_documents(kb_doc, embeddings, collection_name=index_id,
                                  persist_directory=persist_directory, ids=ids)

    """kb_db1 = Chroma(collection_name=index_id, embedding_function=embeddings,
                    persist_directory=persist_directory)
    ##Ram'de kalıyor ?
    x = kb_db1.add_documents(kb_doc)
    print(x)"""
    kb_db1.persist()
    #kb_db1.add_documents()
    #kb_db1.update_document()
    return jsonify({'message': 'Document added successfully'})


@app.route("/indexes/<index_id>/query", methods=["GET"])
def query(index_id):
    local_directory = "kb-h2o-wave-3"
    persist_directory = os.path.join(os.getcwd(), local_directory)
    embeddings = OpenAIEmbeddings()
    kb_db1 = Chroma(collection_name=index_id, embedding_function=embeddings,
                    persist_directory=persist_directory)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    kb_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=kb_db1.as_retriever(),
        return_source_documents=True
    )
    chat_history = []
    query_statement = "What is this document about?"

    # ------ Question Loop ------
    while query_statement != 'exit':
        query_statement = input('Enter your question here: > ')
        if query_statement != 'exit':
            if len(chat_history) > 4090:
                print("Chat history deleted because length is more than 4090,\nCurrent Length:%d", len(chat_history))
                chat_history.pop(0)
                # Update chat_history with the current query and response
            result = kb_qa({"question": query_statement, "chat_history": chat_history})
            # pprint(result["source_documents"])
            pprint(result["answer"])
            pprint(result)
            pprint(chat_history)
    print("---------- Exiting the GPT----------------")
    response = make_response('Exiting the GPT', 200)
    return response
#Is there a way to let Wave know that more than one OpenID authentication option is available for the user? For example, I would like to allow the user to authenticate via Google or ORCID.


@app.route("/base", methods=["GET"])
def base():
    h2o_ai_wave_urls = [
        "https://github.com/h2oai/wave/releases",
        "https://wave.h2o.ai/docs/installation",
        "https://wave.h2o.ai/docs/getting-started",
        "https://wave.h2o.ai/docs/examples",
        "https://github.com/h2oai/wave/issues/693",
        "https://github.com/h2oai/wave/blob/master/.github/CONTRIBUTING.md#development-setup",
        "https://github.com/h2oai/wave/discussions/1897",
        "https://github.com/h2oai/wave/discussions/1888",
        "https://github.com/h2oai/wave/discussions/1885",
        "https://github.com/h2oai/wave/discussions/1865"

    ]

    collection_name = "h2o_wave_knowledgebase-1"
    local_directory = "kb-h2o-wave-3"
    persist_directory = os.path.join(os.getcwd(), local_directory)

    loader = UnstructuredURLLoader(urls=h2o_ai_wave_urls)
    kb_data = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    kb_doc = text_splitter.split_documents(kb_data)

    embeddings = OpenAIEmbeddings()
    kb_db = Chroma.from_documents(kb_doc, embeddings, collection_name=collection_name,
                                  persist_directory=persist_directory)

    kb_db.persist()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    kb_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=kb_db.as_retriever(),
        return_source_documents=True
    )
    chat_history = []
    query_statement = "What is this document about?"

    # ------ Question Loop ------
    while query_statement != 'exit':
        query_statement = input('Enter your question here: > ')
        if query_statement != 'exit':
            if len(chat_history) > 4090:
                print("Chat history deleted because length is more than 4090,\nCurrent Length:%d", len(chat_history))
                chat_history.pop(0)
                # Update chat_history with the current query and response
            result = kb_qa({"question": query_statement, "chat_history": chat_history})
            # pprint(result["source_documents"])
            pprint(result["answer"])
            pprint(result)
    print("---------- Exiting the GPT----------------")
    return 'Web App with Python Flask!'


@app.route("/indexes/<index_id>/remove/<path:url>", methods=["DELETE"])
def remove_documents(index_id, url):
    local_directory = "kb-h2o-wave-3"
    persist_directory = os.path.join(os.getcwd(), local_directory)
    loader = UnstructuredURLLoader(urls=[url])
    kb_data = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    kb_doc = text_splitter.split_documents(kb_data)
    embeddings = OpenAIEmbeddings()
    ids = [f'{i}-{url}' for i in range(1, len(kb_doc) + 1)]
    kb_db1 = Chroma.from_documents(kb_doc, embeddings, collection_name=index_id,
                                  persist_directory=persist_directory, ids=ids)
    print(ids)
    kb_db1.delete(ids=ids)

    return jsonify({"message": "Documents deleted successfully"})


app.run(host='0.0.0.0', port=81)
