# affinity-propagation

按照scikit-learning的AffinityPropagation进行的C++改写。

Demo编译和执行:

```
g++ main.cpp AffinityPropagation.cpp -o ap.bin -I ./ --std=c++11
./ap.bin
```

经过测试，这个C++版本的AffinityPropagation和scikit-learning的结果比较相近，但又不是完全相同。
