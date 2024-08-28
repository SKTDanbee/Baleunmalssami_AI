import socket
import json
import re
import torch
import threading
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.chat_models import ChatOpenAI
from kiwipiepy import Kiwi
import os
import requests

Kiwi = Kiwi()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

api_key = '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'
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
child_id  = "email"

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('/Users/hojoonkim/GITHUB/TEST/텍스트_윤리검증_모델/results/checkpoint-30000')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-bert-multitask')
label_columns = [
    'is_immoral', 'Complaint', 'Favor', 'Impressed', 
    'Fed_up', 'Gratitude', 'Sadness', 'Anger', 'Respect', 'Expectation', 
    'Arrogance', 'Disappointment', 'Determination', 'Distrust', 
    'Satisfaction', 'Comfort', 'Interest', 'Affectionate', 
    'Embarrassment', 'Terror', 'Despair', 'Pathetic', 'Repulsion', 
    'Annoyance', 'Speechless', 'None', 'Defeat', 'Boredom', 
    'Exhaustion', 'Excitement', 'Realization', 'Guilt', 
    'Hatred', 'Delight', 'Confusion', 
    'Shock', 'Reluctance', 'Bitterness', 'Boring', 
    'Pity', 'Surprise', 'Happiness', 'Anxiety', 'Joy', 
    'Trust', 'Discrimination', 'Hate', 'Censure', 'Violence', 
    'Crime', 'Sexual', 'Abuse'
]

# Move model to device
model.to(device)

current_text = ""
lock = threading.Lock()  # Lock for thread safety

def gpt_answer(text, llm):
    llm = llm
    string_template = """
    '{input_text}'가 도덕적인 말이라면 그대로 출력하고, 비도덕적이라면 말을 순화해서 바꿔줘. 
    바뀐 문장만 출력해줘.
    
    ex) '싫어 미친넘아' -> '싫어, 정말 짜증나'
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(string_template) # 고정 입력
    human_message_prompt = HumanMessagePromptTemplate.from_template('{input_text}') # 바뀌는 입력
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt]) # 1,2 합쳐서 GPT에 입력
    
    prompt = chat_prompt.format_prompt(input_text=text).to_messages()
    result = llm.invoke(prompt[0].content)
    
    json_data = {}
    json_data["llm_empty"] = False
    json_data["llm_answer"] = result.content
    
    return json_data

    

def bert_text(text):
    inputs = tokenizer(text)
    inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in inputs.items()}  # Move inputs to GPU
    outputs = model(**inputs)
    preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()  # Detach tensor before converting to NumPy
    json_data = {}
    
    max_emtion_idx = preds[0][1:-6].argmax() + 1
    print(max_emtion_idx )
    json_data['llm_empty'] = True
    json_data['emotion'] = label_columns[max_emtion_idx]
    json_data['is_immoral'] = float(preds[0][0])
    json_data['Discrimination'] = float(preds[0][-7])
    json_data['Hate'] = float(preds[0][-6])
    json_data['Censure'] = float(preds[0][-5])
    json_data['Violence'] = float(preds[0][-4])
    json_data['Crime'] = float(preds[0][-3])
    json_data['Sexual'] = float(preds[0][-2])
    json_data['Abuse'] = float(preds[0][-1])
    
    print(f"Abuse:{json_data['Abuse']},Is_immoral:{json_data['is_immoral']}")
    
    return json_data

def text_input(current_text, json):
    cursor = json['cursor']
    if json['type'] == 'commit':
        char = json['char']
        updated_text = current_text[:cursor] + char + current_text[cursor:]
    elif json['type'] == 'delete':
        if cursor > 0:
            updated_text = current_text[:cursor-1] + current_text[cursor:]
        else:
            updated_text = current_text
    return updated_text

# 문장 종료 판단
def sentence_spliter(current_text, kiwi=Kiwi):
    sentence = kiwi.split_into_sents(current_text)[-1].text
    return sentence
    
def response_text(current_text, child_id):
    report_text = current_text
    end_point = f"https://ansim-app-f6abfdhmexe8ged3.koreacentral-01.azurewebsites.net/create_txt/?report_text={report_text}&child_id={child_id}"
    response = requests.post(end_point)
    print(response.text)
       
    

def handle_client(client_socket):
    global current_text
    try:
        data = client_socket.recv(1024).decode('utf-8')  # 더 큰 데이터 수신 허용
        if not data:
            return
        
        json_data = None
        
        try:
            json_data = re.sub(r'^[^{]*', '', data)
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            print("Received data is not valid JSON")
            return  # 잘못된 데이터를 받았을 때 루프 계속
        
        if json_data:
            if json_data['type'] == 'llm':
                print("llm_requested")
                with lock:
                    llm_answer = gpt_answer(current_text, llm)
                print(llm_answer)
                response_bytes = json.dumps(llm_answer).encode('utf-8')
                client_socket.sendall(response_bytes)
                return
            
            elif json_data['type'] == 'End':
                print("End_requested")
                with lock:
                    response_text(current_text, child_id)
                    current_text = ""
    
            else:
                if json_data['isEmpty'] == True:
                    print("text_empty")
                    with lock:
                        response_text(current_text, child_id)
                        current_text = ""
                with lock:
                    current_text = text_input(current_text, json_data)
                print(current_text)
                return_json = bert_text(current_text)
                try:
                    response_bytes = json.dumps(return_json).encode('utf-8')
                    client_socket.sendall(response_bytes)  # 바이트 데이터 전송
                except Exception as e:
                    print(f"Failed to send data to client: {e}")
    finally:
        client_socket.close()


# 서버 설정
HOST = '0.0.0.0'  # 모든 IP에서 접속 허용
PORT = 12345      # 사용할 포트 번호

# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print(f'Server listening on {HOST}:{PORT}')

while True:
    client_socket, addr = server_socket.accept()
    client_thread = threading.Thread(target=handle_client, args=(client_socket,))
    client_thread.start()
