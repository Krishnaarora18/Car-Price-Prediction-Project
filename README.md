## 📊 Project Structure

- **`notebooks/`** - Jupyter notebooks showing the complete data science process
  - Data exploration and analysis
  - Feature engineering and preprocessing
  - Model training and evaluation
- **`data/`** - Both Raw and Cleaned Datasets are available in the notebooks folder
- **`flask_app.py`** - Flask web application for price predictions
- **`templates/`** - Beautiful dark-themed HTML interface

## 🔬 Data Science Process

1. **Data Exploration** (`notebooks/car-price.ipynb`)
   - Dataset analysis and visualization
   - Feature correlation studies
   - Outlier detection and handling

2. **Model Development**
   - StandardScaler preprocessing
   - One-hot encoding for 35+ car brands
   - Model training and validation

3. **Web Deployment**
   - Flask application with dark theme UI
   - Real-time predictions
   - Deployment is currently being debugged. Local setup works as expected. Contribution welcome!

     
### Frontend Note
The initial version of the HTML/CSS frontend was generated with assistance from Claude sonnet 4.0(Anthropic AI). It was further customized and integrated with the Flask backend.

