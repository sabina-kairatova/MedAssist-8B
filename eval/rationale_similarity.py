from rouge import Rouge
import pandas as pd


def rouge_score(model_output, gt_output):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(model_output, gt_output)
    return rouge_scores[0]['rouge-l']['f']


def read_reference(file_path):
    data_list = []
    df = pd.read_parquet(file_path, engine='auto')
    for _, row in df.iterrows():
        question = row["question"].strip()
        options = ", ".join([f"{key}: {value}" for key, value in row["options"].items()])
        gt_answer = row["answer"]
        gt_reasoning = row["explanation"]
        prompt_dict = {
                "prompt": f"""The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\"\n{question} {options}\n""",
                "gt_answer": gt_answer,
                "gt_reasoning": gt_reasoning
        }
        data_list.append(prompt_dict)
    return data_list


def calculate_score(model_output_file, reference_file):
    references_list = read_reference(reference_file)

    with open(model_output_file, "r") as fp:
        text = fp.read()
    
    output_list = text.split("[SPLIT]")
    
    score_list = []
    for ref, output in zip(references_list, output_list):
        try:
            model_reasoning = output.split(ref["prompt"])[1]
        except:
            continue
        model_reasoning = model_reasoning.split("Ground truth answer: ")[0] if len(model_reasoning.split("Ground truth answer: ")) > 1 else model_reasoning
        gt_reasoning = ref["gt_reasoning"]
        score_list.append(rouge_score(model_reasoning, gt_reasoning))

    return sum(score_list) / len(score_list)
    

if __name__ == "__main__":
    reference_path = "./eval/op5_test.parquet"
    finetuned_output_path = "./eval/medbullet_output_finetuned.txt"
    basemodel_output_path = "./eval/medbullet_output_basemodel.txt"
    finetuned_rouge_score = calculate_score(finetuned_output_path, reference_path)
    basemodel_rouge_score = calculate_score(basemodel_output_path, reference_path)
    print("Finetuned Model Performance")
    print(f"Rationale similarity score on MedBullet Benchmark (Rouge-L F1-score): {finetuned_rouge_score}")
    print("Base Model Performance")
    print(f"Rationale similarity score on MedBullet Benchmark (Rouge-L F1-score): {basemodel_rouge_score}")
    