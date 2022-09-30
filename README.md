# py_lens
Rust based python package to apply a wave distortion filter to an image.

<table>
<tr>
<td>Base image</td>
<td>apply_lens(image, cx=0, cy=0, u=25)</td>
<td>apply_lens(image, cx=1, cy=0.5, u=100)</td>
</tr>
<tr>
<td>
<img src="images/G1.jpg">
</td>
<td>
<img src="images/G1-0-0-25.jpeg">
</td>
<td>
<img src="images/G1-1-05-100.png">
</td>
</tr>
<tr>
<td>Base image without modification <br> Created in GIMP</td>
<td>Wave starts in top left (0,0 position) with 25x strength</td>
<td>Wave starts in the lower centre position (900px&nbsp;down, 450px&nbsp;across) with a 100x&nbsp;strength modifier</td>
</tr>

</table>


### Building with maturin
` maturin build --bindings cffi `

### Running in python

```python
import py_lens
import cv2
import numpy

image: numpy.ndarray = cv2.imread(r"/images/G1.jpg", cv2.IMREAD_COLOR)
image_wave = py_lens.apply_lens(image, cx=1, cy=0.5, u=100)
cv2.imshow('G1-1-05-100.jpeg', image_wave)
cv2.waitKey()
```
