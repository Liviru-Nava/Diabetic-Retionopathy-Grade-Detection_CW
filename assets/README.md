# assets/

Static images used by the DRxVision web app.

| File | Used for |
|---|---|
| `drxvision_icon_retina.png` | The app's icon, shown in the browser tab and in the page header |
| `sample_photos_by_grade.jpg` | Example photos of each grade, shown on the "Dataset exploration" tab. It is a smaller JPEG copy of `outputs/figures/01_data_exploration/sample_photos_by_grade.png`, so the app loads quickly |
| `drxvision_icon_monogram.png` | Alternative icon design, not currently used by the app |
| `drxvision_icon_scan.png` | Alternative icon design, not currently used by the app |

To change the app icon, point `APP_ICON_FILE` near the top of `app.py` at a different file in this folder.