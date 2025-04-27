@echo off
echo Installing Python dependencies...

:: Install dependencies one by one
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install xgboost==2.0.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install ucimlrepo==0.0.3

echo Installation complete!
pause 