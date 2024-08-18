import socket
import json
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set device to GPU if available
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./텍스트_윤리검증_모델/results/checkpoint-30000')
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

def bert_text(text):
    inputs = tokenizer(text)
    inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in inputs.items()}  # Move inputs to GPU
    outputs = model(**inputs)
    preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()  # Detach tensor before converting to NumPy
    json_data = {}
    
    max_emtion_idx = preds[0][1:-6].argmax() + 1
    print(max_emtion_idx )
    json_data['emotion'] = label_columns[max_emtion_idx]
    json_data['is_immoral'] = float(preds[0][0])
    json_data['Discrimination'] = float(preds[0][-7])
    json_data['Hate'] = float(preds[0][-6])
    json_data['Censure'] = float(preds[0][-5])
    json_data['Violence'] = float(preds[0][-4])
    json_data['Crime'] = float(preds[0][-3])
    json_data['Sexual'] = float(preds[0][-2])
    json_data['Abuse'] = float(preds[0][-1])
    
    return json_data

def text_input(text_list, json):
    if json['type'] == 'commit':
        cursor = json['cursor']
        text = json['char']
        text_list.insert(cursor-1, text)
        return text_list
    elif json['type'] == 'delete':
        cursor = json['cursor']
        if cursor == 0:
            return text_list
        try:
            text_list.pop(cursor-1)
        except IndexError:
            print(f"Warning: Index {cursor-1} is out of bounds for text_list with size {len(text_list)}")
        return text_list

# 서버 설정
HOST = '0.0.0.0'  # 모든 IP에서 접속 허용
PORT = 12345      # 사용할 포트 번호

# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))

# 클라이언트 연결 대기
server_socket.listen(5)
print(f'Server listening on {HOST}:{PORT}')

text_list = []

while True:
    client_socket, addr = server_socket.accept()
    with client_socket:
        data = client_socket.recv(4096).decode('utf-8')  # 더 큰 데이터 수신 허용
        if not data:
            break
        
        json_data = None
        
        try:
            json_data = re.sub(r'^[^{]*', '', data)
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            print("Received data is not valid JSON")
            continue  # 잘못된 데이터를 받았을 때 루프 계속
        
        if json_data:
            text_list = text_input(text_list, json_data)
            print("".join(text_list)) 
            print(client_socket)
            return_json = bert_text("".join(text_list))
            try:
                # client_socket.connect(("192.168.0.86",12346))
                # client_socket.sendmsg([b"Hello"])
                
                response_bytes = json.dumps(return_json).encode('utf-8')
                client_socket.sendall(response_bytes)  # 바이트 데이터 전송
                client_socket.close()
            except Exception as e:
                print(f"Failed to send data to client: {e}")


