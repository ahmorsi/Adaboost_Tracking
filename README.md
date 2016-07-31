Build Instructions :-

```
mkdir build
cd build
cmake ..
make
```

Run Instructions :-

```
cd nemo/
../bin/nemo_tracking  <training-file> <test-file> <# iterations for AdaBoost>
```

Run Example :-

```
../bin/nemo_tracking  frames.train frames.test 50
```

![alt tag](http://imgur.com/LXbD6dU.png)
