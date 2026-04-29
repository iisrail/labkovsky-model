@echo off
REM Upload model weights and ChromaDB to Modal volumes
REM Run this before deploying

echo === Uploading Labkovsky model to Modal ===

echo Uploading model weights...
modal volume put labkovsky-model-weights models/vikhr-labkovsky-awq /model --force
if errorlevel 1 goto :error

echo Uploading ChromaDB...
modal volume put labkovsky-chroma-db chroma_db /chroma_db --force
if errorlevel 1 goto :error

echo === Upload complete ===
echo.
echo Now deploy with: modal deploy modal_deploy.py
goto :end

:error
echo Upload failed!
exit /b 1

:end
