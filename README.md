# speech-recognition

This repository presents a lab activity focused on speech recognition. The objective of the task is to recognize the pronunciation of a single word from a list of 35 words using a multilayer perceptron (MLP). The lab activity utilizes the Speech Commands Data Set, which consists of 105,829 recordings of the 35 words, divided into training, validation, and test sets. 

## Dataset and Preprocessing

The Speech Commands Data Set has already undergone feature extraction, resulting in spectrograms that have been made uniform in size. The dataset is partitioned into training, validation, and test sets to facilitate model training and evaluation.

## Lab Components

The lab activity encompasses several components, including:

1. Spectrogram Visualization: The visualization of spectrograms provides an understanding of the acoustic characteristics of the speech data.

2. Feature Normalization: Feature normalization techniques are applied to preprocess the data, enhancing the model's ability to learn and generalize.

3. MLP Training: A multilayer perceptron model without hidden layers is trained on the preprocessed data.

4. Network Architecture Exploration: Different network architectures are explored to analyze their impact on the model's performance.

5. Performance Evaluation: A confusion matrix is constructed to summarize the MLP's behavior, allowing insights into its performance. Classification errors are further analyzed to identify patterns and potential areas of improvement.

## Parameter Analysis

To gain a comprehensive understanding of the model's performance, the experiments are replicated using different feature normalization techniques, batch sizes, and lambda values. These variations help in evaluating how these parameters influence the model's accuracy and effectiveness.

## Contact

If you have any questions or suggestions regarding this project, please feel free to contact me via email at davide.ligari01@gmail.com

Thank you for your interest in this project!
