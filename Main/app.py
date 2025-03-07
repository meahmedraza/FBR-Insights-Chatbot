import streamlit as st

from transformers import AutoTokenizer , AutoModelForCausalLM , BitsAndBytesConfig
from peft import get_peft_model , LoraConfig
import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st 

def extract_answer(context) : 

    context = context[context.find('Answer' , 10) :]
    context = context[: context.find('Question')]

    return context 
def get_completion(query , model , tokenizer) :

    embeds = tokenizer(query , return_tensors = 'pt' , add_special_tokens = True)
    model_inputs = embeds.to('cuda:0')

    generated_ids = model.generate(
        **model_inputs ,
        do_sample = True ,
        max_new_tokens = 100 ,
        pad_token_id = tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(generated_ids[0] , skip_special_tokens = True)

    return decoded

def answer(question , model , tokenizer) : 

    prompt = '''
    Answer the following questions, based on the provided context only, do not hallucinate 

    Context : {}

    Question : {}
    '''

    with st.spinner('Loading VectorStore') : 

        embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
        vc = FAISS.load_local('vc' , embeddings = embeddings , allow_dangerous_deserialization = True)

        similar_docs = vc.similarity_search(question)

        context = ' '.join([
            element.page_content
            for element 
            in similar_docs
        ])

    with st.spinner('Getting answer from the Model') : 

        result = get_completion(
            query = prompt.format(context , question) , 
            model = model ,
            tokenizer = tokenizer)

        result = extract_answer(result)

    return result



torch.cuda.empty_cache()

from huggingface_hub import login
login('Enter your huggingface token here')

def format_path(path) : 

    path = path.replace('\n' , '')
    path = path.replace('\t' , '')
    path = path.replace(' ' , '')

    return path

with st.spinner('Loading Tokenizer and Model') : 
    tokenizer = AutoTokenizer.from_pretrained(
        format_path(
            '''
            meta-llama/
                Llama-2-7b-hf
            '''
        ) , 
        padding = True , 
        truncation = True , 
        token ='Enter your huggingface token here')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        format_path(
            '''
            meta-llama/
                Llama-2-7b-hf
            '''
        ) , 
        device_map = {'' : 0} ,
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True ,
            bnb_4bit_compute_dtype = torch.float16 ,
            bnb_4bit_quant_type = 'nf4' ,
            bnb_4bit_use_double_quant = False 
        ))

    lora_config_ = LoraConfig.from_pretrained('llama2_instruct')
    model_tuned = get_peft_model(model, lora_config_)

def check_prompt(prompt) : 

    try : 
        prompt.replace('' , '')
        return True 
    except : return False


def check_mesaage() : 
    '''
    Function to check the messages
    '''

    if 'messages' not in st.session_state : st.session_state.messages = []

check_mesaage()

for message in st.session_state.messages : 

    with st.chat_message(message['role']) : st.markdown(message['content'])

prompt = st.chat_input('Ask me anything')

if check_prompt(prompt) :

    with st.chat_message('user') : st.markdown(prompt)

    st.session_state.messages.append({
        'role' : 'user' , 
        'content' : prompt
    })

    if prompt != None or prompt != '' : 

        response = answer(question = prompt , model = model_tuned , tokenizer = tokenizer)

        with st.chat_message('assistant') : st.markdown(response)

        st.session_state.messages.append({
            'role' : 'assistant' , 
            'content' : response
        })
