
# Knowledge Distillation on DeBERTa-V3-large with GLUE Tasks

## Overview
This repository contains the implementation of a knowledge distillation task using the DeBERTa-V3-large model, specifically applied to the GLUE benchmark tasks. Our approach is based on the DeBERTa-v3 repository, which you can explore here: [DeBERTa-v3 Repository](https://github.com/microsoft/DeBERTa).

## Getting Started

We include a detailed pipeline for the CoLA task. 

1. LLM Replacement Test on CoLA:
   ```bash
   ssh cola_llm.sh
   ```
1. Knowledge Distillation on CoLA:
   ```bash
   ssh cola_kd.sh
   ```

For other tasks, such as SST-2, MRPC, or QNLI, you can follow the same steps, ensuring to update the task name and data paths accordingly.

## Dataset Preparation

Since all the datasets is too large, we don't include them in our repo. You can download the GLUE dataset from ./experiments/glue 
Then, you can copy the dataset to ./gather_json/ for further processing.