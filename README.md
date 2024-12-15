# Course Description

> Advanced topics in building and verifying software systems, selected from areas of current research such as: model checking and automated verification, testing and automated test generation, program synthesis, runtime verification, machine learning and its applications in the design of verified systems, formal analysis of machine learning algorithms, principles of programming languages and type systems.

[Source](https://siebelschool.illinois.edu/academics/courses/cs521-120248)

# HW1 Adversarial Attacks & Training

Adversarial attacks and training on neural networks. Implementation of the Projected Gradient Descent attack described
in the [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083) paper.

# HW2 Interval Analysis for Neural Networks

# HW4

`adversarial_explanations/`: SmoothGrad explanations for adversarial examples.

`explanations/`: LIME and SmoothGrad explanations for normal examples.

`output`: correlation scores, adversarial images

# HW5

## Problem 1

Problem 1 Google Colab [Notebook](https://colab.research.google.com/github/trakyari/Trustworthy-AI-Systems/blob/main/hw5/problem1.ipynb)

### To run

Activate your hugging face token

```bash
export HUGGINGFACE_TOKEN=your_token
```

## Problem 2

Not fully working. Data loading and model training functionality is present.

Problem 2 Google Colab [Notebook](https://colab.research.google.com/github/trakyari/Trustworthy-AI-Systems/blob/main/hw5/problem2.ipynb)

## Problem 3

### To run

Set your OpenAI API key in a file `key.txt` in the same directory as the notebook.

Download the VQA Balanced Real Images questions, images, and complimentary pairs from the [VQA Balanced Real Images](https://visualqa.org/download.html) page.

Place the downloaded files in the `data` directory.

Unzip the files in the `data` directory.

Problem 3 Google Colab [Notebook](https://colab.research.google.com/github/trakyari/Trustworthy-AI-Systems/blob/main/hw5/problem3.ipynb)
