# Image Retrieval System

A content-based image retrieval (CBIR) pipeline that compares classical and learned
approaches to finding visually similar images. Given a query image, the system searches
an indexed database and returns the closest visual matches — using several different
feature representations and matching strategies, benchmarked against each other.

## What's implemented

- **SIFT** keypoint/descriptor extraction, with keypoints visualized on sample images
- **CNN features** from a pre-trained VGG16 — both the last convolutional layer and the
  final fully-connected layer, compared for retrieval suitability
- **Cosine similarity** and **Euclidean distance**, implemented from scratch (no
  `scipy`/`sklearn` metric calls) and compared on retrieval quality
- A full **bag-of-visual-words** pipeline: SIFT descriptors → k-means codebook → each
  image represented as visual-word IDs → inverted index for fast lookup
- **Vector database integration** with [Pinecone](https://www.pinecone.io/) for scalable
  nearest-neighbor search over CNN embeddings
- Quantitative evaluation with **mean Average Precision (mAP)** and precision-recall
  curves, plus a qualitative look at retrieval failure cases

Dataset: [Imagenette](https://github.com/fastai/imagenette), a 10-class subset of
ImageNet (auto-downloaded by the notebook).

## Results

| Method | mAP |
|---|---|
| CNN embeddings (cosine similarity) | **0.733** |
| SIFT bag-of-visual-words | 0.128 |

The CNN embedding approach substantially outperforms the classical SIFT/BoVW pipeline for
category-level retrieval — CNN features encode semantic content, while SIFT matches
low-level local patches and is much weaker at telling categories apart. The notebook
includes precision-recall curves for both methods and a discussion of failure cases (e.g.
queries where background similarity outweighs object identity).

## Getting started

```bash
pip install -r requirements.txt
```

The Imagenette dataset is downloaded automatically by the first cell of the notebook.

The vector-database section uses [Pinecone](https://www.pinecone.io/) — create a free
account, grab an API key, and set it up locally:

```bash
cp .env.example .env
# then edit .env and paste in your own PINECONE_API_KEY
```

Everything except the Pinecone section will run without any account setup.

Run `notebooks/image_retrieval.ipynb` top to bottom.

## Project structure

```
image-retrieval-system/
├── notebooks/
│   └── image_retrieval.ipynb   # all the code, results, and analysis
├── requirements.txt
├── .env.example
└── README.md
```

## Tech stack

Python · PyTorch/torchvision (VGG16) · OpenCV (SIFT) · scikit-learn (k-means) · Pinecone · matplotlib
