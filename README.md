---
title: HybridMedicalVision  
emoji: 🏥  
colorFrom: blue  
colorTo: green  
sdk: gradio  
sdk_version: "4.16.0"  
app_file: app.py  
pinned: false  
license: MIT license  
---

# 🏥 HybridMedicalVision

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/Adi3003/medical-ct-xray-analyzer)

A hybrid AI system for detecting anomalies in chest X-rays using advanced deep learning.

## Features

**Medical Imaging Analysis**
- Chest X-ray classification (Normal / Pneumonia detection)    
- High-accuracy predictions with confidence scoring  
- Support for medical screening workflows  

**General Object Detection**
- YOLOv8-based detection for other images  
- Multi-object identification  
- Detailed confidence reports  

## Model Details

- **Architecture**: DenseNet121 with medical pre-training  
- **Training Data**: 200,000+ chest X-rays from multiple datasets  
- **Test Accuracy**: 87.84%  
- **Processing**: GPU-accelerated with model caching  
- **Device Support**: Auto-detection (CUDA / MPS / CPU)  

## Usage

1. Upload an image (X-ray, or general image)  
2. Click the **Analyze** button  
3. Review the classification results and confidence scores  

## Technical Stack

- **Deep Learning**: PyTorch  
- **Model**: DenseNet121  
- **Object Detection**: YOLOv8  
- **Interface**: Gradio  
- **Image Processing**: OpenCV, Albumentations  

## Performance

- X-Ray Classification Accuracy: **87.84%**  
- Average Inference Time: \< 1 second (GPU)  
- Model Size: ~50 MB  
- Support for: JPG, PNG, BMP formats  

## Medical Disclaimer

This system is provided for **educational and research purposes only**.

- **NOT a medical device**  
- Results must be verified by qualified healthcare professionals  
- Always consult with licensed medical professionals for medical decisions  
- Do not rely solely on this system for clinical diagnosis  

## Privacy & Security

- Images are **NOT stored** or used for training  
- Processing happens **locally on the server**  
- No data is collected or transmitted outside the application  

## Author

**Aditya Rai**

## License

MIT License – See LICENSE file for details

## Support

For issues, suggestions, or questions, please open an issue on the repository.

---

*Last Updated: 2025*
