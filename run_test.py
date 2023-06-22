import argparse

from transformers import T5ForConditionalGeneration, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path", dest="model_path", action="store", default="./post_output_save")
    args = parser.parse_args()

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    results = []

    with open('./testset.txt', 'r', encoding='utf-8') as testsets:
        while True:
            line = testsets.readline()
            if not line: break

            tests = line.split(',')
            age = int(tests[0].strip()) // 10
            sex = 0 if tests[1].strip() == '남' else 1
            question = tests[2].strip().replace('\n', '')

            prompt = f'dialogue_a{age}_g{sex} : {question}'
            tokens = tokenizer.encode(prompt, return_tensors='pt').cuda()

            output_tokens = model.generate(
                tokens,
                max_length = 256,
                no_repeat_ngram_size = 1,
                do_sample=True,
                temperature=0.4,
                top_p=0.95,
                top_k=3,
                num_return_sequences=1,
                early_stopping=True,
            )

            for output in output_tokens:
                decodes = tokenizer.decode(output, skip_special_tokens=True)
                results.append(f'{prompt} -> {decodes}')
        testsets.close()

    with open('./test_result.txt', 'w+', encoding='utf-8') as outputs:
        for result in results:
            outputs.write(result + '\n')
        outputs.flush()
        outputs.close()

if __name__ == "__main__":
    main()
