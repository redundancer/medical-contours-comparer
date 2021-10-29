# medical-contours-comparer
Compare medical contours with a bunch of different metrics.

# Install requirements
```
pip install -r requirements.txt
```

# Help
For help run
```
python3 compare_contours.py -h
```

# Folder structure
The structure of the root folder must be

```
[root_dir]-
          |--/[patient_folderx]-----
          |                        |-[*gt-substring*.*]
          |                        |-[*pred-substring*.*]
          |
          |--/[next_patient_folder]-
                                   |-[*gt-substring*.*]
                                   |-[*pred-substring*.*]
          ...
```
# Execute
To execute the script run
```
python3 compare_contours.py --root_dir path/to/root_dir --gt substring1 substring2 --pred substring3 substring4
```

In case that each two contours are of different shape, the 'groud truth' contour will be resampled to the 'predicted' contour. This project obviously was created to compare manually and artificially created 3D medical contours.

If everything worked as expected, a folder 'results' will be created, with a subfolder named as your 'root_dir' folder. In there two files will be created, namely 'summarized_results.csv' and 'detailed_results.csv'.
