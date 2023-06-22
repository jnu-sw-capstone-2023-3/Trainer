# Trainer
모델 학습을 위한 트레이너 및 학습 데이터

# npc_train.py
: 모델 학습을 위한 파이썬 코드입니다.
game_dialogues에 대화 csv 데이터셋, unlabeled_data에 비정형 데이터셋을 넣어 작동시킵니다.

# run_test.py
: 모델 테스트를 위한 파이썬 코드입니다.
학습이 완료된 모델을 불러와 testset.txt에 있는 질문들에 대답하여 test_result.txt에 출력합니다.
질문은 '나이, 성별, 질문'의 구조로 각 라인을 파싱합니다.

# TransportNet.py
: Flask를 이용해 모델을 실행시켜두고, HTTP GET REQUEST를 통해 입력을 받아 결과를 Response로 출력합니다.


