# Create a Synthetic Multivariate Time Series Data of At Least 1 GB and Apply Anomaly Detection Algorithms  

## Introduction  
This project was part of my internship application process for **AI4SEE PRIVATE LIMITED** through **Internshala**. The assignment was to generate a **synthetic multivariate time series dataset** of at least 1 GB, apply anomaly detection algorithms, and present the results effectively.  

The challenge, although intimidating, turned out to be one of my most rewarding learning experiences, filled with moments of trial and error, small wins, and a steep learning curve. Below, I document the entire process, including the tools and strategies used, as well as insights gained.  

---

## Assignment Instructions  
The assignment details, as provided, were as follows:  

> Create a synthetic multivariate time series data of at least 1 GB using the provided code. Apply any simple anomaly detection algorithm and display the results. Please share the dataset link.  
---

## My Experience  

When I started, I underestimated the challenge. The **initial dataset I generated was only 10 MB**, far from the 1 GB requirement. I adjusted parameters, experimented with different techniques, and gradually increased the dataset size:  

1. **Step 1**: My first attempt resulted in **10 MB** of data, which felt discouraging but gave me a baseline.  
2. **Step 2**: By tweaking the code and increasing iterations, I generated **128 MB**, a significant improvement but still not sufficient.  
3. **Step 3**: After multiple iterations and experiments, I **finally crossed the 1 GB mark**, achieving **1.44 GB** of synthetic data!  

This process tested my patience and problem-solving skills. It required creativity to balance performance, computation time, and data size.  

### Technical Hurdles  

- **Processing Time**: My system, powered by an **NVIDIA GeForce RTX 4050**, took around **40 minutes** to generate the full dataset.  
  - On lower-end systems, this could take significantly longer (1+ hours).  
- **Memory Management**: Handling such a large dataset required careful planning to avoid crashing my system.  
- **Graph Generation**: Organizing and visualizing the results in a clear and meaningful way was crucial.  

---

## Requirements  

Before running the code, ensure the following libraries are installed:  

```bash
pip install wmi psutil pandas
```  

These libraries are essential for generating the dataset and visualizing results.  

---

## Steps to Run the Project  

1. **Dataset Generation**  
   - Adjust the parameters in the provided code as needed. Larger parameters will produce larger datasets, but they will also increase processing time.  
   - My settings resulted in **1.44 GB** of data, which took **40 minutes** to process.  

2. **Anomaly Detection**  
   - Apply any simple anomaly detection algorithm to analyze the dataset.  
   - I used statistical techniques to detect outliers and present the results visually.  

3. **Visualization**  
   - Generate well-organized graphs to illustrate the anomalies and data trends effectively.  

---

## Results  

The project generated:  

- A **1.44 GB synthetic multivariate time series dataset**.  
- Anomaly detection results with clear and meaningful graphs.  

The results were successfully presented in a manner aligned with the requirements.  

---

## Key Insights  

1. **Experimentation is Key**: The journey from 10 MB to 1.44 GB taught me the importance of persistence and iterative refinement.  
2. **System Optimization**: Adjusting parameters like loop iterations and data density was critical for managing computation time and resource usage.  
3. **Anomaly Detection**: Implementing and visualizing a simple algorithm effectively communicated the results.  

---

## Final Note  

For faster results on low-end PCs, reduce the parameter values in the code. This will decrease the dataset size but also significantly cut down the processing time. However, achieving a 1+ GB dataset may require additional runtime.  

Thank you for taking the time to read about my project journey! If you have any questions or need further clarifications, feel free to reach out.  

---

## And here are the results
You can access the generated dataset using the following link:  
[Dataset Link](https://drive.google.com/file/d/1pRpOHhChlq1MU1bde_lS7b15mddLESZ1/view?usp=drive_link)
Image is also uploaded as Figure1 in files here
