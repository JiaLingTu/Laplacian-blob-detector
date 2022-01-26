# Laplacian-blob-detector-
Implementing Laplacian blob detector without well-developed library function, and do not complete by y calling cv2.SimpleBlobDetector(),  cv2.GaussianBlur(), cv2.Laplacian(), etc.

## Requirements
Setup packages
```sh
pip install -r requirements.txt
```

## Descriptions
Run main.py file using the following command lines
```sh
--s 2 --sigma 1.8 --num_octave 4 --threshold_rel 0.9 --img butterfly.jpeg
```
Or you can modify the code `detector = blob_detector(Opt)` to `detector = blob_detector(params)`  

> Note: Test image should be in img folder.
