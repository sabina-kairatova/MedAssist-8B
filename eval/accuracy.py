def load_output(file_path: str):
    output_list = []
    with open(file_path, "r") as fp:
        text = fp.read()
    output = text.split("[SPLIT]")
    for i, item in enumerate(output):
        answer = item.split("Therefore, the answer is ")[1][1] if len(item.split("Therefore, the answer is ")) > 1 else None
        gt_answer = item.split("Ground truth answer: ")[1][0] if len(item.split("Ground truth answer: ")) > 1 else None
        output_list.append({"Question #": i, "Answer": answer, "GT_answer": gt_answer})
    return output_list

def score(output_list):
    score = 0
    for output in output_list:
        if output["Answer"] == output["GT_answer"]:
            score += 1
    return score / len(output_list) * 100

if __name__ == "__main__":
    medqa_output = load_output("./eval/medqa_output.txt")
    medbullet_output = load_output("./medbullet_output.txt")
    print(f"Test Accuracy Score for 'MedQA Benchmark': {score(medqa_output)}")
    print(f"Test Accuracy Score for 'MedBullet-5op Benchmark': {score(medbullet_output)}")

