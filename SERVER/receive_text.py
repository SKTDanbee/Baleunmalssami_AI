# 소켓 수신
import socket
import json
import re
# 문자 입력 처리
def text_input(text_list, json):
    if json['type'] == 'commit':
        cusor = json['cursor']
        text = json['char']
        
        text_list.insert(cusor-1, text)
        return text_list
    
    elif json['type'] == 'delete':
        cusor = json['cursor']
        if cusor == 0:
            return text_list
        text_list.pop(cusor-1)
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
    # 클라이언트 연결 수락
    client_socket, addr = server_socket.accept()
    # 클라이언트로부터 데이터 수신
    data = client_socket.recv(1024)
    if not data:
        break
    
    json_data = None
    
    try:
        json_data = data.decode('utf-8')
        # { 앞에 있는 모든 문자 제거
        json_data = re.sub(r'^[^{]*', '', json_data)
        json_data = json.loads(json_data)
    except json.JSONDecodeError:
        print("Received data is not valid JSON")
    
    text_list = text_input(text_list, json_data)
    print("".join(text_list))
    # 클라이언트에게 응답 보내기
    client_socket.sendall(b'Hello from server!')

    # 연결 종료
    client_socket.close()
