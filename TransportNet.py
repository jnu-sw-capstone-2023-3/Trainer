import argparse

from flask import Flask, request
from transformers import T5ForConditionalGeneration, AutoTokenizer

app = Flask(__name__)
model, tokenizer = None, None


@app.route('/', methods=['GET'])
def get_request():
    query_params = request.args

    tokens = tokenizer.encode(query_params.get('text'), return_tensors='pt')

    output_tokens = model.generate(
        tokens,
        max_length=128,
        no_repeat_ngram_size=1,
        do_sample=True,
        temperature=0.4,
        top_p=0.95,
        top_k=3,
        num_return_sequences=1,
        early_stopping=True,
    )

    result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-port", "--port", dest="port", action="store", default=8888)
    parser.add_argument("-mp", "--model_path", dest="mp", action="store", default="./post_output_save")
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.mp)
    tokenizer = AutoTokenizer.from_pretrained(args.mp)

    app.run(debug=True, port=int(args.port))