# DINOv3 Image Search Tool

**[日本語](README.md) | English**

![Screenshot](screenshot.png)

An image search application using DINOv3 (Vision Transformer). Provides a beautiful GUI with PyQt6 and enables similar image search using image features.

## Features

- **DINOv3 Model**: Uses Facebook's latest self-supervised learning model
- **Image Search**: Search for similar images from an image
- **Database Management**: Persist features with SQLite
- **Folder/ZIP Processing**: Batch image registration
- **Beautiful GUI**: Modern UI with PyQt6
- **Drag & Drop**: Easy operation

## Installation

### Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended, but works with CPU)

### Install Dependencies

```bash
pip install -r requirements_dinov3.txt
```

## Usage

### 1. Launch the Application

```bash
python dinov3_search_en.py
```

The DINOv3 model will be automatically downloaded on the first launch.

### 2. Add Images

#### Method 1: Drag & Drop
- Drag and drop image files to the image display area

#### Method 2: Select from Buttons
- Click "Select Image" button
- Click "Select Folder" button for batch registration
- Click "Select ZIP" button to batch register images in ZIP

### 3. Image Search

1. Drop or select an image
2. Similar images are automatically searched and displayed on the right
3. Color-coded display by similarity (0-100%)

### 4. Database Management

- **Create**: Create a new database
- **Clear DB**: Clear the contents of the current database
- **Delete DB**: Completely delete the database file
- **Refresh**: Update the database list

## Main Features

### Image Search
- Search for similar images from an image
- Ranking display by cosine similarity
- Display top 10 similar images

### Database
- Save features with SQLite
- Duplicate check (MD5 hash)
- Automatic thumbnail generation

### Batch Processing
- Batch register images in a folder
- Batch register images in a ZIP file
- Progress display

## Technical Specifications

### DINOv3 Model
- Model: `facebook/dinov2-base`
- Input size: 224x224
- Features: 768 dimensions (CLS token)

### Database Schema

#### images table
- id: Image ID (primary key)
- file_path: File path
- file_hash: MD5 hash
- thumbnail: Thumbnail image (BLOB)

#### features table
- image_id: Image ID (foreign key)
- feature_vector: Feature vector (BLOB)

## Differences from CLIP

### DINOv3 Features
- ✅ Specialized in image-only feature extraction
- ✅ Trained with self-supervised learning
- ✅ High-quality image features
- ❌ Text search not supported

### CLIP Features
- ✅ Supports both images and text
- ✅ Can search images from text
- ✅ Multimodal learning

## Troubleshooting

### Model download fails
- Check internet connection
- Check access to Hugging Face

### Feature extraction is slow
- Check if GPU is available
- Check if CUDA is installed correctly

### Out of memory error
- Reduce image size
- Process in smaller batches during batch processing

## License

This project is released under the MIT License.

## References

- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Hugging Face - DINOv2](https://huggingface.co/facebook/dinov2-base)
