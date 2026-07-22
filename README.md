# Human Activity Recognition Dissertation

## Project Overview

This repository contains the implementation developed as part of the **MSc Software Engineering** dissertation at the **University of West London**.

The project investigates the effectiveness of transfer learning-based Convolutional Neural Network (CNN) architectures for **Image-Based Human Activity Recognition (HAR)** through a comparative study of **ResNet** and **MobileNet** under consistent experimental conditions.

---

## Dissertation Title

**A Comparative Study of Transfer Learning-Based CNN Architectures for Image-Based Human Activity Recognition**

---

## Author

**Ujjwal Shrestha**  
MSc Software Engineering  
University of West London

---

## Features

This project includes:

- Dataset analysis
- Data preprocessing
- Transfer learning using ResNet and MobileNet
- Model training
- Performance evaluation
- Comparative analysis

---

## Prerequisites

Before setting up the project, ensure the following software is installed:

- Python 3.11.x
- Git
- Visual Studio Code (recommended)

---

## Clone the Repository

### Using SSH (Recommended)

```bash
git clone git@github.com:ujjwal-27/HAR_dissertation.git
cd HAR_dissertation
```

### Using HTTPS

```bash
git clone https://github.com/ujjwal-27/HAR_dissertation.git
cd HAR_dissertation
```

---

## Create a Virtual Environment

```bash
python3.11 -m venv .venv
```

---

## Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows (Command Prompt)

```cmd
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify the Installation

Check the Python version:

```bash
python --version
```

Expected output:

```text
Python 3.11.x
```

---

## Dataset

This project uses the **Stanford40** dataset for image-based Human Activity Recognition.

The dataset is **not included** in this repository due to licensing and size constraints.

Once obtained, organise the dataset as follows:

```text
HAR_dissertation/
│
├── dataset/
│   ├── images/
│   └── ImageSplits/
```

Further dataset preparation instructions will be provided as the implementation progresses.

---

## Project Structure

The project will be developed incrementally. Directories and modules will be added as required while maintaining a clean and modular architecture.

```text
HAR_dissertation/
│
├── dataset/
├── src/
├── notebooks/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Licence

This repository is intended for academic and research purposes as part of an MSc dissertation at the University of West London.