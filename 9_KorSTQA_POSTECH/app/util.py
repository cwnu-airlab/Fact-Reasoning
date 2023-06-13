import json 

def load_korean_passage(passage_path):
    with open(passage_path, 'r') as f: 
        passage_list = json.load(f)
    
    doc_to_document_passages = dict()
    doc_title_to_context = dict()
    mypassage_list = []
    for passage_title, passage_content in passage_list.items(): 
        passage_content["title"] = passage_title 
        content = passage_content["content"]
        doc_to_document_passages[content] = passage_content 
        doc_title_to_context[passage_title] = content
        mypassage_list.append(passage_content)
    
    return mypassage_list, doc_to_document_passages, doc_title_to_context

