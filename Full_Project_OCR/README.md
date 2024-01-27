# Optical Character Recognition Project
This repository is part of my journey in learning AI and Machine Learning with dibimbing.id. The project focuses on Optical Character Recognition (OCR) using mmocr, specifically leveraging the dbnet model for text detection and the svtr model for text recognition. The goal is to automatically identify each word from the student's answers.

## Project Overview

### Case Scenario
As a learner on dibimbing.id, I encounter scenarios where I need to process and analyze student answers. The task involves:
1. Text Detection: Utilizing the dbnet model to automatically detect text regions within the student's response.
2. Text Recognition: Employing the svtr model to recognize the text content within the identified regions.

### Evaluation
To assess the accuracy of the OCR results, I utilize the Levenshtein distance metric. This allows me to compare the OCR-generated text with the ground truth, providing insights into the effectiveness of the models.

## Components and Tools
1. Programming Language: [Python Logo](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png)
2. OCR Toolbox: mmocr - An open-source toolbox for OCR tasks.
3. Recognition Models:
4. dbnet - for text detection
5. svtr - for text recognition
6. Annotation Tool: Label-Studio - Facilitating the creation of ground truth data for training and evaluation.
7. Ground Truth: The reference data against which OCR results are compared for evaluation.

## How to Use
1. Clone the Repository:
'''git clone https://github.com/your-username/ocr-project.git'''
2. Install Dependencies:
'''pip install -r requirements.txt'''
3. Run the OCR Pipeline:
Execute the main OCR script to process student answers, perform text detection, recognition, and generate results.
4. Evaluate Results:
Utilize the evaluation script to compare OCR results with the ground truth, calculating the Levenshtein distance for each response.

