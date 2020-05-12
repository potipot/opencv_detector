
Specify OpenCV installation path in 
```
./opencv_dnn/CMakeLists.txt
```

Build with
```
cmake --DCMAKE_INSTALL_PREFIX /usr/bin/cmake Debug .
make
```

Run with 
```
./opencv_detector /path/to/images/directory
```
Note: images are searched recursively
