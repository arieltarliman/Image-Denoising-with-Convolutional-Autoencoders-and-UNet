# Image Denoising using Convolutional Autoencoders and U-Net Architectures

This repository presents a deep learning project focused on image denoising. The core task is to build and evaluate neural network models capable of removing artificially added Gaussian noise from images, restoring them to their original, clean state. The project compares a baseline Convolutional Autoencoder against more advanced U-Net-based architectures.

## **Project Overview**

In real-world applications, images are often corrupted by noise during acquisition or transmission, which can degrade visual quality and hinder the performance of subsequent computer vision tasks. This project tackles this issue by developing deep learning models that learn to reverse the noise corruption process.

The main challenge was to work with a dataset that did not initially contain noisy images. Therefore, a key part of the project involved creating a noisy dataset by adding Gaussian noise (with a mean of 0.0 and a standard deviation of 0.1) to clean images. The models were then trained to take these noisy images as input and produce clean, denoised images as output.

## **Methodology**

The project explores the effectiveness of autoencoder-based models for image reconstruction. An autoencoder is a type of neural network trained to learn a compressed representation (encoding) of its input and then reconstruct the input from that encoding. For denoising, the model is trained with noisy images as input and the original clean images as the target output.

### **1. Model Architectures**

Three different architectures were implemented and compared:

* **Baseline Convolutional Autoencoder (CAE)**: A standard autoencoder model consisting of an encoder and a decoder. The encoder uses a series of convolutional and max-pooling layers to downsample the image into a compressed latent representation. The decoder then uses convolutional and upsampling layers to reconstruct the image from this representation.

* **Modified U-Net Autoencoder**: This model is based on the U-Net architecture, which is renowned for its success in biomedical image segmentation. It features a similar downsampling (contracting) path as the CAE but includes "skip connections" that concatenate the output of encoder layers with the input of corresponding decoder layers. These connections allow the decoder to access high-resolution features from the encoder, helping to preserve fine details during reconstruction.

* **Tuned U-Net Autoencoder**: An enhanced version of the Modified U-Net where hyperparameters (like learning rate) and architectural details were fine-tuned to optimize performance.

### **2. Training and Evaluation**

All models were trained to minimize the reconstruction error between the model's output and the ground truth (clean) images.

* **Optimizer**: Adam optimizer.
* **Loss Function**: Binary Crossentropy was used to measure the pixel-wise difference between the reconstructed and original images.
* **Evaluation Metric**: The **Structural Similarity Index (SSIM)** was used as the primary evaluation metric. SSIM is a perception-based metric that measures the similarity between two images based on luminance, contrast, and structure. A value closer to 1 indicates a higher similarity to the original clean image, signifying better denoising performance.

## **Results and Analysis**

The models were evaluated both quantitatively using the loss and SSIM metrics and qualitatively through visual inspection of the denoised images.

### **Quantitative Comparison**

The U-Net based models significantly outperformed the baseline autoencoder.

**Model Loss Comparison**
*The training and validation loss curves show that both U-Net models achieved a much lower loss compared to the baseline, indicating a more accurate reconstruction.*
![Model Loss Comparison](img/model%20loss%20comparison.png)

**Model SSIM Metric Comparison**
*The SSIM scores confirm the superiority of the U-Net models. They achieved a much higher structural similarity to the ground truth images, demonstrating their effectiveness in preserving important image details.*
![Model SSIM Metric Comparison](img/model%20ssim%20metric%20comparison.png)

### **Qualitative (Visual) Comparison**

The visual results clearly illustrate the performance difference between the models.

**Denoised Sample Comparison**
*The baseline model's output is blurry and loses significant detail. In contrast, both U-Net models produce remarkably sharp and clear reconstructions that are almost indistinguishable from the ground truth.*
![Comparison Sample Output](img/comparison%20sample%20output.png)

### **Conclusion**

The results unequivocally demonstrate that the **U-Net architecture is vastly superior** to the baseline convolutional autoencoder for this image denoising task. The inclusion of skip connections allows the U-Net models to retain crucial high-frequency details that are lost in the baseline model's bottleneck. Both the modified and tuned U-Net models produced excellent, high-fidelity results, effectively removing noise while preserving the structural integrity and texture of the original images.

## **How to Get Started**

To run this project, you will need to set up a Python environment with the necessary deep learning libraries.

### **Prerequisites**

* Python 3.x
* TensorFlow
* NumPy
* Matplotlib

### **Installation & Execution**

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your_username/Image-Denoising-with-Convolutional-Autoencoders-and-UNet.git](https://github.com/your_username/Image-Denoising-with-Convolutional-Autoencoders-and-UNet.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd Image-Denoising-with-Convolutional-Autoencoders-and-UNet
    ```
3.  **Install the required packages:**
    ```sh
    pip install tensorflow numpy matplotlib
    ```
4.  **Run the notebook:**
    Launch Jupyter Notebook and open the `AutoEncoder.ipynb` file to see the complete workflow, from data preparation and model building to training and evaluation.
    ```sh
    jupyter notebook AutoEncoder.ipynb
    ```
