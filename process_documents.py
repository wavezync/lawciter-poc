import os
from langchain.prompts import SystemMessagePromptTemplate
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI
import uuid

# set openai api key
os.environ["OPENAI_API_KEY"] = "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"


def load_document(file):
    # print(type(file))
    if type(file) == str:
        with open(file, 'r', encoding='utf-8') as f:
            file_contents = f.read()
            return file_contents
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    elif file.type == "application/pdf":
        # create a temp file and load it
        fname = os.path.join("temp", f"{str(uuid.uuid4())}.pdf")
        with open(fname, "wb") as f:
            f.write(file.read())
        data = UnstructuredPDFLoader(file_path=fname).load()
        os.remove(fname)
        return data
    else:
        return None


def build_prompts(template):
    prompt_templates = template['prompt_templates']
    context = prompt_templates['context']
    instructions = prompt_templates['instructions']
    output_format = prompt_templates['output_format']

    context_prompt = SystemMessagePromptTemplate.from_template(
        f"""
        Note: ±±± is a special token that indicates the start and end of a instruction.

        You have given following instructions:
        
        ±±±
        {context}
        ±±±
        """)
    instructions_prompt = SystemMessagePromptTemplate.from_template(
        f"""
        Follow these instructions strictly
        You need to go thorough the document and extract the following information. If none found, leave it blank.
        ±±±
        {instructions}
        ±±±

        """)
    output_format_prompt = SystemMessagePromptTemplate.from_template(f"""
        Please output in markdown format.
        Use following example as a reference. 

        ±±±                                                                                                                                          
        {output_format}
        ±±±
    """)

    return context_prompt, instructions_prompt, output_format_prompt


def process_document(doc, template):
    context_prompt, instructions_prompt, output_format_prompt = build_prompts(
        template)
    loaded_doc = load_document(doc)

    if loaded_doc is None:
        return "Invalid document type"

    prompts = (context_prompt.format() +
               instructions_prompt.format(input=loaded_doc) + output_format_prompt.format())
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k')

    return llm.predict(prompts.format())
