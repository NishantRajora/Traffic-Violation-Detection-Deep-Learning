@echo off
title Traffic Violation Video Detection
echo ========================================================
echo   Traffic Violation Video Detection - YOLO11 Object Detection
echo ========================================================
echo.
set KMP_DUPLICATE_LIB_OK=TRUE
python videodetection.py
pause
