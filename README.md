# Intelligent Image Processing Tool

The **Intelligent Image Processing Tool** is a Python-based application that provides a suite of image processing features, including noise addition, noise removal, and edge detection. It is designed for users who need an easy-to-use interface to experiment with various image processing techniques.

## 🎨 Features

- **Upload Images**: Simple interface to upload and view images.
- **Add Noise**: Add different types of noise to images, including:
  - Gaussian Noise
  - Salt-and-Pepper Noise
  - Poisson Noise
  - Speckle Noise
  - Uniform Noise
  - Periodic Noise
  - Quantization Noise
- **Noise Removal**: Restore the original image easily with a reset option.
- **Edge Detection**: Detect edges in images using custom-implemented algorithms:
  - Canny
  - Sobel
  - Laplacian
- **User Interface**: Built using `tkinter` to provide an intuitive, graphical way of interacting with the tool.

## 📜 Core Algorithms

The algorithms for noise addition, noise removal, and edge detection are all implemented from scratch. This includes the following:

- **Noise Types**: Various noise models like Gaussian, Salt-and-Pepper, and Speckle are applied programmatically.
- **Edge Detection**: Includes custom-built implementations of popular edge detection algorithms like Canny, Sobel, and Laplacian.
  
Libraries like `OpenCV` are only used for basic tasks such as image reading and display, while the processing algorithms themselves are fully coded without relying on external implementations.

## 📂 Project Structure

The project is organized in a modular way:
- `main.py`: Entry point of the application with the GUI.
- `noise_addition.py`: Contains all the noise generation algorithms.
- `edge_detection.py`: Contains edge detection algorithms.
- `utils.py`: Helper functions for image handling.

---

Developed by **Osama Alkhatib**.
