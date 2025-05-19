# Breast Cancer Detection Project

This project focuses on developing machine learning models for breast cancer detection using medical imaging datasets. The project currently supports two major datasets:

1. CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
2. BreakHis (Breast Cancer Histopathological Database)

## Project Structure

```
breastcancer/
├── data/                  # Data directory (gitignored)
│   ├── CBIS_DDSM/        # CBIS-DDSM dataset
│   └── processed_images/  # Preprocessed images
├── ExtractCBISDDSM.py    # Script for downloading and preprocessing CBIS-DDSM
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Prerequisites

- Python 3.8 or higher
- TCIA API key (for CBIS-DDSM dataset)
- Kaggle account (for BreakHis dataset)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/breastcancer.git
cd breastcancer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Setup

### TCIA API Key (for CBIS-DDSM)
1. Register at [TCIA Portal](https://www.cancerimagingarchive.net/tcia-portal/access-tcia/)
2. Get your API key
3. Set the API key as an environment variable:
```bash
# Windows
set TCIA_API_KEY=your_api_key
# Linux/Mac
export TCIA_API_KEY=your_api_key
```

### Kaggle Setup (for BreakHis)
1. Create a Kaggle account
2. Download your API credentials (kaggle.json)
3. Place the kaggle.json file in the `.kaggle` directory

## Usage

### CBIS-DDSM Dataset
To download and preprocess the CBIS-DDSM dataset:

```python
from ExtractCBISDDSM import extract_cbis_ddsm_data

# Download and preprocess 5 series
metadata = extract_cbis_ddsm_data(max_series=5)
```

The script will:
1. Download DICOM images from TCIA
2. Convert them to PNG format
3. Normalize pixel values
4. Save metadata to CSV

## Data Processing

The project includes preprocessing steps for medical images:
- DICOM to PNG conversion
- Pixel value normalization
- Metadata extraction and organization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [TCIA](https://www.cancerimagingarchive.net/) for the CBIS-DDSM dataset
- [Kaggle](https://www.kaggle.com/) for the BreakHis dataset 