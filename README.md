# 🤖 FBR Insight Chatbot (LLM + RAG)

This repository contains the **FBR Insight Chatbot**, a project built using **Large Language Models (LLM)** and **Retrieval-Augmented Generation (RAG)**.  
The chatbot is designed to answer queries related to **Federal Board of Revenue (FBR) data**, making use of fine-tuned models, experiment tracking, and knowledge retrieval pipelines.

---

## 📌 Features
- 💬 **Interactive chatbot** for FBR-related queries.  
- ⚙️ **RAG (Retrieval-Augmented Generation)** pipeline for accurate, context-aware responses.  
- 🧠 **LoRA fine-tuning** on domain-specific data.  
- 📊 Visualization of workflows and architecture diagrams.  
- 🔍 Support for PDF/document-based question answering.  
- 📈 **Experiment tracking with Weights & Biases (wandb)** for model training and fine-tuning.  

---

## 📂 Repository Structure

Main/ # Core project files

├── app.py # Main application entry (Streamlit chatbot app)

├── requirements.txt # Required dependencies

├── cleaning.py # Data cleaning script

├── extractor.py # Information extraction module

├── generator.py # Response generation pipeline

├── train.py # Training script (fine-tuning with wandb integration)

├── fbrinsight/ # Core chatbot package

├── Notebooks/ # Jupyter notebooks for experiments

│ ├── llama3-seq-2-seq.ipynb

│ ├── processing.ipynb

├── Images/ # Supporting diagrams/screenshots

│ ├── detailed work-flow.jpeg

│ ├── Sequence Diagram.png

│ ├── Use Case Diagram.png

│ ├── Lora Config.png

│ ├── processflow.png

│ ├── output test.png

│ └── instruction fintuning.png

├── Questions.pdf # Sample test questions

├── Test Questions.docx # Testing document

└── Readme.md # Project documentation

---

## 🖼️ Workflow & Diagrams
The repository includes:  
- ✅ **Workflow Diagrams** (`flow.jpeg`, `detailed work-flow.jpeg`)  
- ✅ **Use Case Diagram** (`Use Case Diagram.png`)  
- ✅ **Sequence Diagram** (`Sequence Diagram.png`)  
- ✅ **LoRA Fine-tuning Configurations**  

---

## 🛠️ Tech Stack
- Python  
- Hugging Face Transformers  
- LangChain  
- FAISS (vector database for RAG)  
- LoRA (Low-Rank Adaptation for fine-tuning)  
- Streamlit (frontend for chatbot)  
- **Weights & Biases (wandb)** – training & experiment tracking  

---

## 📊 Experiment Tracking with Weights & Biases
- All model fine-tuning and evaluation experiments were logged using **[wandb](https://wandb.ai/)**.  
- Provides **loss curves, accuracy metrics, hyperparameter tracking**, and training logs.  
- Enables easy comparison between **baseline models vs fine-tuned models**.  

---

# 🖥️ Demo / Sample Output

Here’s a sample chatbot output screenshot:  

![Chatbot Output](fbrinsight/Images/'output test.png')


## 🚀 Deployment Instructions

Here are the insturctions to deploy the model on your side, verify if you ahve NVIDIA GPU, else the model will not work 

Make sure you have an **NVIDIA GPU** available.  
Without GPU, the model will not run properly due to heavy computation requirements.

1. Open a terminal/cmd inside the project folder.  
2. Install dependencies:
   pip install -r requirements.txt


then wait for some seconds installation to complete.

3. Run the Streamlit app:
* type 
```
streamlit run app.py
```

The model and tokenizer takes a little bit longer time to load for the first time 

Open the provided URL in your browser.

⚡ Note: The model and tokenizer may take some time to load for the first run.


# 📄 Documentation

Questions.pdf – FBR domain sample queries.
Test Questions.docx – Used for chatbot evaluation.

# 🎯 Purpose

This project was developed as part of an academic initiative to apply LLM fine-tuning & RAG techniques for solving real-world problems. Specifically, it aims to help users query and understand FBR policies, processes, and datasets efficiently.




