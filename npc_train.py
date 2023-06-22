import argparse
import csv
import json
import os.path
import re

from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, TrainingArguments, Trainer


class Data:
    def get_text(self):
        pass

class DialogueData(Data):
    def __init__(self, age, gender, text, answer):
        self.age = age
        self.gender = gender
        self.text = text
        self.answer = answer

    def get_text(self):
        return f'dialogue_a{self.age}_g{self.gender} : {self.text}', self.answer

class TextData(Data):
    def __init__(self, text):
        self.text = text

    def get_text(self):
        return self.text

class TrainDataset(Dataset):
    def __init__(self, tokenizer, datas:list):
        self.datas = datas
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        value = self.datas[item]
        ret = {'input_ids': None, 'attention_mask': None, 'labels': None}
        if type(value) is DialogueData:
            q, a = value.get_text()
            qtoken = self.tokenizer(q, max_length=256, padding='max_length', return_tensors='pt', truncation=True)
            atoken = self.tokenizer(a, max_length=256, padding='max_length', return_tensors='pt', truncation=True)
            ret['input_ids'] = qtoken.input_ids.squeeze(0)
            ret['attention_mask'] = qtoken.attention_mask.squeeze(0)
            ret['labels'] = atoken.input_ids.squeeze(0)
        else:
            tokens = self.tokenizer(value.get_text(), max_length=256, padding='max_length', return_tensors='pt', truncation=True)
            ret['input_ids'] = tokens.input_ids.squeeze(0)
            ret['attention_mask'] = tokens.attention_mask.squeeze(0)
            ret['labels'] = tokens.input_ids.squeeze(0)
        return ret

    def __len__(self):
        return len(self.datas)

def parse_attr(text):
    conversion = {
        'gender': ['gender', '성별'],
        'age': ['age', '나이', '연령'],
        'from': ['origin', '출신', '원산지', 'Hometown', '고향', '기원', 'region of origin', '출신 지역', '출신지역', '출신지', 'hometown', '기원 지역'],
        'now': ['region', '지역', 'current location', '현재', '현 위치', '현 거주지', 'current', '현재 위치', '현재위치', '현재 거주지',
                'current residence', 'currently living', '현생', '현', 'current region', '현재 지역', '현재지', '현재지역', '현 지역', '현지', '현재 사는 곳', '현재 거주', '거주지'],
        'race': ['race', '인종', '종족']
    }

    find = re.findall(r'[a-zA-Z가-힣]*: [a-zA-Z가-힣0-9]*', text)
    attr = {}
    for parse in find:
        key, value = parse.split(': ')
        for conv_key, values in conversion.items():
            if key.lower() in values:
                if conv_key == 'age':
                    attr[conv_key] = int(value[:1])
                elif conv_key == 'gender':
                    gender_value = 0 if value == 'male' or '남' in value else 1
                    attr[conv_key] = gender_value
                break

    return attr

def prestudy(model, tokenizer, dataset, epochs, batch_size, gas):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=256, padding='max_length', return_tensors='pt')

    train_args = TrainingArguments(
        adafactor=True,
        output_dir='./pre_output',
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gas,
        num_train_epochs=epochs,

        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=2000,
        warmup_ratio=0.02,

        save_steps=1000,
        save_total_limit=3,
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    path = './pre_output_save'
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    return model, tokenizer

def poststudy(model, tokenizer, dataset, epochs, batch_size, gas):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=256, padding='max_length', return_tensors='pt')

    train_args = TrainingArguments(
        output_dir='./post_output',
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gas,
        num_train_epochs=epochs,

        learning_rate=5e-4,
        weight_decay=0.1,

        save_steps=500,
        save_total_limit=3,
        logging_steps=300,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    path = './post_output_save'
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gd", "--game_dialogue", dest="gd", action="store", default='./game_dialogues')
    parser.add_argument("-pd", "--public_dialogue", dest="pd", action="store", default='./public_data')
    parser.add_argument("-ud", "--unlabeled_data", dest="ud", action="store", default='./unlabeled_data')
    parser.add_argument("-mp", "--model_path", dest="model_path", action="store", default='KETI-AIR/ke-t5-base')
    parser.add_argument("-pdr", "--public_data_ratio", dest="pdr", action="store", default=0.3)
    parser.add_argument("-be", "--pre_epoch", dest="pre_epoch", action="store", default=1)
    parser.add_argument("-ae", "--post_epoch", dest="post_epoch", action="store", default=20)
    parser.add_argument("-smr", "--span_masking_ratio", dest="smr", action="store", default=0.2)
    parser.add_argument("-batch", "--batch_size", dest="batch_size", action="store", default=2)
    parser.add_argument("-acc", "--gradient_accumulation", dest="grad", action="store", default=8)
    parser.add_argument("-dpt", "--do_pre_train", dest="do_pre_train", action="store", default=True)
    args = parser.parse_args()

    loaded = 0
    dialogues = []
    unlabels = []

    # Game Dialogues is CSV
    if os.path.exists(args.gd):
        print(f'Game Dialogue Datas load from {args.gd}')
        for file in os.listdir(args.gd):
            path = f'{args.gd}/{file}'
            with open(path, 'r', encoding='utf-8') as file_open:
                csv_data = csv.reader(file_open)
                for data in csv_data:
                    split = data[1].split('\n')
                    question = re.sub(r'[a-zA-Z][0-9]*\. |[\'\",]', '', split[0])
                    split2 = split[1].split(')', 1)
                    attributes = parse_attr(split2[0])
                    answer = split2[1].strip()

                    if not re.match(r'[가-힣]', answer):
                        continue # Skip English only data

                    if 'age' in attributes.keys() or 'gender' in attributes.keys():
                        dialogues.append(DialogueData(attributes['age'], attributes['gender'], question, answer))
        print(f'Load {len(dialogues)} Dialogues!')
        loaded += len(dialogues)

    # Unlabeled Data is JSON ( TEXT Array )
    if os.path.exists(args.ud):
        print(f'Unlabeled Datas load from {args.ud}')
        for file in os.listdir(args.ud):
            path = f'{args.ud}/{file}'
            with open(path, 'r', encoding='utf-8') as file_open:
                file_data = json.load(file_open)
                for fd in file_data:
                    unlabels.append(TextData(fd))
        print(f'Load {len(unlabels)} Datas!')

    # Public Diaglogue is JSON
    if os.path.exists(args.pd):
        categories = 12
        dialogue_cat = [round(len(dialogues) * float(args.pdr) / categories)] * categories
        category_key = {}
        print(f'Public Dialogue Datas load from {args.pd} ( ratio {args.pdr} )')

        for file in os.listdir(args.pd):
            path = f'{args.pd}/{file}'
            with open(path, 'r', encoding='utf-8') as file_open:
                file_data = json.load(file_open)
                for dialogue in file_data:
                    question = dialogue['question']
                    answer = dialogue['answers']
                    tk = answer['age'] * 10 + answer['gender']
                    if tk not in category_key.keys():
                        category_key[tk] = len(category_key)
                    tk = category_key[tk]

                    if dialogue_cat[tk] > 0 and '<' not in question['text'] and '<' in answer['text']:
                        dialogues.append(DialogueData(answer['age'], answer['gender'], question['text'], answer['text']))
                        dialogue_cat[tk] -= 1
        print(f'Load {len(dialogues) - loaded} Dialogues!')

    print(f'Ready to study. ( Dialogues : {len(dialogues)}, Unlabeled : {len(unlabels)} )')

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_tokens(['<sigh>', '<laugh>', '<sad>'])

    dpt = False if args.do_pre_train == "False" else True
    print(f'PRE TRAIN : {dpt}')

    print(f'Model Size : {model.num_parameters()}')

    if dpt:
        model, tokenizer = prestudy(model, tokenizer, TrainDataset(tokenizer, unlabels), int(args.pre_epoch), int(args.batch_size), int(args.grad))
    poststudy(model, tokenizer, TrainDataset(tokenizer, dialogues), int(args.post_epoch), int(args.batch_size), int(args.grad))

if __name__ == "__main__":
    main()


# 3882
# 3654
# 3287
# 3290
# 3441
# 3786
# 3272
# 3663
# 3294
# 3452
# 3502.1
