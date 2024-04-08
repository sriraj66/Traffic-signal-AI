# Problem Statement
For legal professionals, staying updated with the latest laws and judgments can be a challenging task. According to LexusNexus research, nearly 65 percent of a lawyer's time is dedicated solely to legal research on relevant laws. Additionally, grappling with lengthy verdicts and chargesheets is a time-consuming and arduous process. The preparation of legal documents is also a taxing task that demands meticulous attention. In response to these challenges, we have undertaken a project to develop a smart Legal Language Model (LLM) fine-tuned on legal data, capable of addressing the aforementioned issues. This article will provide an in-depth exploration of our project, highlighting key components and technologies that have facilitated the development of an effective solution.

# Description
We employed the Zephyr-7b-beta model, surpassing many larger models of its kind in terms of performance. Despite its enhanced capabilities, controlling its proclivity for hallucinations proved to be challenging. Extensive training was conducted using a substantial synthetic dataset gathered from diverse platforms, including large language model completion datasets, open-source information, and legal databases. This exhaustive training equipped the model with comprehensive knowledge of Indian laws, recent developments, significant judgments, and more.

# Post training Quantization ![image](https://github.com/sriraj66/Traffic-signal-AI/assets/75485469/2574a6b1-9b54-470c-a444-24eb0633768d)
Leveraging a large Language Model for inference poses challenges, and the OpenVINO Neural Network Compression Framework (NNCF) method for quantization proves to be an excellent solution. The detailed steps outlined in notebook 254, available in the training folder, were instrumental for post-training quantization. In a trial run, we applied this method to the actual Zephyr 7b beta model without further training. The model was successfully converted to INT8 format using only the CPU, resulting in a streamlined 6 GB bin-sized model. This transformation significantly accelerated the inference process without any discernible drop in performance and later in discord, it is stated that usage of Openvino toolkit is prohibited, so this model is not utilized and code used for quantizing the model in IDC is attached.

# Experiments:
Various models, including Llama2, Flang T5, Mistral 7b, and Zephyr 7b, were explored for summarization and data generation. Despite encountering several challenges, the Zephyr 7b model emerged as the preferred choice due to its superior performance compared to other models of similar size.

# Usecase of Intel Developer cloud
The Intel Developer Cloud proves to be an excellent platform, offering access to powerful CPUs and high-speed internet, thereby facilitating a remarkably swift process. This challenges the misconception that LLM training necessitates GPU usage. The experimentation phase demonstrated that faster inferencing and training are achievable with different models on this platform.

For our misfortune at last moment when we trained the model with actual data, it got disconnected, which made us not use it at present, and the codes and screenshots are attached and the model is trained on other platform as per suggestion of intel team.

![image](https://github.com/sriraj66/Traffic-signal-AI/assets/75485469/ddbbc853-fea6-4e7f-b628-13de9982fe9d)

# Future scope

1.To enhance performance through increased computational capacity, we aim to construct an expansive dataset for the aforementioned use cases. This augmented dataset will serve as the foundation for retraining a more robust language model, enabling superior capabilities. The intention is to leverage advanced computing power to refine and elevate the model's proficiency in generating legal documents.

2.We embarked on retraining the model using Intel frameworks and incorporated quantization with NNCF. This approach yielded improved results, showcasing the model's enhanced performance. However, a setback occurred as our session expired, preventing us from saving the valuable progress made during this training endeavor. Despite this challenge, the discernible advancements in model performance underscored the effectiveness of the adopted methodologies.


3.Enhancing the model's hardware versatility, we aim to achieve GPU independence by seamlessly integrating Intel frameworks such as Pandas and NumPy into the frontend. This strategic implementation not only ensures improved efficiency but also contributes to the overall optimization of the application. By fostering compatibility with Intel frameworks at the frontend, we empower the model to operate seamlessly across different hardware configurations, thereby enhancing its accessibility and performance.


4.In order to enhance the functionality of our document creation application, we are incorporating voice capability for an improved user experience. Additionally, we are implementing automatic printing and document validation features to streamline the entire document creation process. This integration aims to provide users with a more efficient and seamless workflow, reducing manual efforts and ensuring the accuracy and completeness of generated documents.

5.To ensure widespread accessibility and future usability, the team is strategically planning to deploy the application in the cloud. This strategic decision aims to provide legal professionals and lawyers across India with seamless access, facilitating widespread utilization of the platform. The deployment in the cloud not only enhances scalability but also fosters collaboration, enabling legal practitioners to leverage the tool efficiently and contribute to the broader legal community.

# Learnings and Insight

1.Specialized NLP Focus:
 - Expanding expertise in NLP, specifically in question answering and text generation.

2.End-to-End Legal Assistant Application:
 - Training the Large Language Model (LLM) for a comprehensive legal assistant application.
 - Enabling simultaneous capabilities for text generation and question answering.

3.Framework Exploration:

 - Investigating different frameworks and fine-tuning methods.
 - Experimenting with models such as Llama2 13b, Flang T5, and Mistral 7b.
 - Identifying compatibility issues, with only the quantized Zephyr 7b model proving suitable for training with the available dataset.

 # Future Application Enhancement for intel:
  - Recognizing the prospective benefits of integrating Intel's features in future iterations.
  - Envisaging heightened end-to-end application performance through the strategic application of recently acquired insights and technologies.
  
  - Currently, our trained models undergo quantization to the GPTQ format for optimized performance, requiring the use of GPUs. Looking ahead, there is a potential shift towards quantizing them to GGML or OV formats, facilitating efficient inferencing even with CPU resources, as an alternative to the     current methodology.

# Tech stack used:
---------------streamlit
---------------IDC Jupyter Notebook Intel AI analytic ToolKit
---------------Model T5 zephyr


# Conclusion:

In conclusion, "Enlightening Justice" not only showcases the transformative power of AI and the Intel OneAPI AI Analytics Toolkit in the legal domain but also highlights the resilience of the team in overcoming challenges. The successful creation of a Smart Legal Companion, Advanced Document Summarization System, and Legal Document Generator underscores the project's positive impact on legal professionals. Despite setbacks, the project's adaptability and forward-thinking approach ensure a promising trajectory for future advancements, marking a significant step toward revolutionizing legal support through cutting-edge technology and innovative solutions.


# Quick Steps

Required installation

```pip install faiss-cpu streamlit langchain huggingface_hub sentence_transformers pypdf peft streamlit_option_menu auto-gptq optimum diffusers```

clone repository

``` 
https://github.com/sriraj66/Traffic-signal-AI
```


# Application

```  run demo.py ```
set the path of demo.py from hackathon folder


