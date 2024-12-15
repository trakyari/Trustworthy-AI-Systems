import csv
import json
import itertools
from sklearn import base
from sympy import comp
from tqdm import tqdm
from openai import OpenAI
import io
from PIL import Image
from torchvision import datasets, transforms
import base64

open_ai_key = ""
with open("key.txt", "r") as file:
    open_ai_key = file.read().strip()
client = OpenAI(api_key=open_ai_key)

base_prompt = "Please answer the following question based on the image and the question.\n"
base_prompt += "The answer should only contain the answer type and nothing else.\n"
base_prompt += "The answer should follow the answer type of "


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSON decode error: {file_path}")
        return None
    return data


def encode_image_to_base64(image_path: str) -> str:
    """
    Opens a JPEG image from the specified path and encodes it in Base64.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    try:
        # Open the image file in binary mode
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Encode the binary data to Base64
        base64_encoded_image = base64.b64encode(image_data)

        # Convert Base64 bytes to a string
        return base64_encoded_image.decode('utf-8')
    except FileNotFoundError:
        return "Error: File not found. Please provide a valid image path."
    except Exception as e:
        return f"Error: {e}"


def prompt_llm_for_vqa(queries, images, answer_type):
    try:
        base64_images = []
        for image in images:
            base64_images.append(encode_image_to_base64(image))

        complete_prompt = base_prompt + answer_type + "\n"
        complete_prompt += "the question: "
        completions = []
        for n in range(2):
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                                "text": complete_prompt + queries[n]},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpg;base64,{base64_images[n]}"},
                            },
                        ],
                    },],
            )
            completions.append(completion.choices[0].to_dict())

        return completions
    except Exception as e:
        print(f"Error processing image: {e}")


def save_completions_to_csv(completions, filename='results.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
                                'question_id_one', 'question_id_two', 'answer_one', 'answer_two'])
        writer.writeheader()
        for completion in completions:
            writer.writerow({
                'question_id_one': completion['question_id_one'],
                'question_id_two': completion['question_id_two'],
                'answer_one': completion['answer_one'],
                'answer_two': completion['answer_two']
            })


def main():
    complementary_pairs = load_json_file(
        "data/v2_mscoco_val2014_complementary_pairs.json")

    if complementary_pairs is None:
        print("Error loading complementary pairs file")
        exit(1)

    questions = load_json_file(
        "data/v2_OpenEnded_mscoco_val2014_questions.json")

    if questions is None:
        print("Error loading questions file")
        exit(1)

    annotations = load_json_file(
        "data/v2_mscoco_val2014_annotations.json")

    if annotations is None:
        print("Error loading annotations file")
        exit(1)

    questions = questions["questions"]

    question_id_to_idx = {}
    question_annotations = {}

    # flatten complementary pairs
    # select first 10 pairs only
    complementary_pairs = complementary_pairs[:10]
    complementary_pairs_flat = list(itertools.chain(*complementary_pairs))
    # convert to hash map
    for comp_pair in complementary_pairs_flat:
        question_id_to_idx[comp_pair] = 1

    for idx, question in enumerate(questions):
        if question["question_id"] in question_id_to_idx and question_id_to_idx[question["question_id"]] == 1:
            question_id_to_idx[question["question_id"]] = idx

    for idx, annotation in enumerate(annotations["annotations"]):
        if annotation["question_id"] in question_id_to_idx:
            question_annotations[annotation["question_id"]] = annotation

    completions = []
    for comp_pair in complementary_pairs:
        queries = []
        images = []
        answer_type = ""
        for question_id in comp_pair:
            idx = question_id_to_idx[question_id]
            queries.append(questions[idx]["question"])
            images.append(
                f"data/val2014/COCO_val2014_{str(questions[idx]['image_id']).zfill(12)}.jpg")

            answer_type = question_annotations[question_id]["answer_type"]

        print(queries)
        print(images)
        print(answer_type)

        completions.append(prompt_llm_for_vqa(queries, images, answer_type))

    results = [
        {
            'question_id_one': comp_pair[0],
            'question_id_two': comp_pair[1],
            'answer_one': completions[0]["message"]["content"],
            'answer_two': completions[1]["message"]["content"]
        }
        for completions, comp_pair in zip(completions, complementary_pairs)
    ]

    save_completions_to_csv(results)

if __name__ == "__main__":
    main()
