# Text Summarizer using T5 and Streamlit

A text summarization web application built using a fine-tuned T5-small model on the SAMSum dataset and deployed using Streamlit. The application generates concise abstractive summaries from dialogue-style conversations.

---

## Features

* Abstractive text summarization
* Fine-tuned T5-small model
* Interactive Streamlit web interface
* Generates concise summaries from long conversations
* End-to-end training and inference pipeline

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Streamlit
* Pandas
* NumPy

---

## Dataset

Dataset Used: SAMSum

SAMSum is a dialogue summarization dataset containing messenger-like conversations paired with human-written summaries.

---

## Project Structure

text-summarizer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── saved_summary_model/
└── notebook/
     └──  text_summarizer.ipynb

## Training Details

Base Model: T5-small

Dataset: SAMSum

Training Notebook:
notebook/text_summarizer.ipynb

---

## Example

Input:

A: Are you coming tomorrow?
B: Yes, I will be there by 10 AM.

Generated Summary:

B will arrive tomorrow at 10 AM.

---

## Future Improvements

* Summary length control
* Batch summarization support
* ROUGE score evaluation
* Deployment on Streamlit Community Cloud
* Support for larger T5 variants

---

## Author

Aman Singh Chauhan

B.Tech CSE Student

GitHub: https://github.com/amansingh0718
