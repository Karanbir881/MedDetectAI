@echo off
REM Runs the image experiments: HealthCure baseline vs improved models.
REM Uses the "meddetect" conda environment if it has TensorFlow, otherwise tf_env.
REM Pass --full to include EfficientNetB0 (much slower on a CPU).
cd /d "%~dp0"
set "PY="
set "CONDA_PY=%USERPROFILE%\miniconda3\envs\meddetect\python.exe"
if exist "%CONDA_PY%" "%CONDA_PY%" -c "import tensorflow" 2>nul && set "PY=%CONDA_PY%"
if not defined PY set "PY=%~dp0tf_env\Scripts\python.exe"
echo Using %PY%
"%PY%" -c "import sklearn, cv2, imutils, scipy" 2>nul || "%PY%" -m pip install scikit-learn opencv-python imutils scipy
if not exist training\results mkdir training\results
set "MODE=--quick"
if /i "%~1"=="--full" set "MODE="
echo Running experiments. Progress is written to training\results\run_log.txt
"%PY%" -u -m training.run_image_experiments --data "C:\Users\Karan\Downloads\meddetect\meddetectAI" %MODE% > training\results\run_log.txt 2>&1
echo Finished. Results: training\results\images.md
pause
