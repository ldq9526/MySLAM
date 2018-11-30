echo "Configuring and building 3rdparty/DBoW2 ..."
cd 3rdparty/DBoW2
if [ ! -d "./build/" ];then
  mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16

cd ../../g2o
echo "Configuring and building 3rdparty/g2o ..."
if [ ! -d "./build/" ];then
  mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16

cd ../../..
echo "Configuring and building ORB_SLAM2 ..."
if [ ! -d "./build" ];then
  mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16

cd ..
echo "Extracting ORBVocabulary ..."
if [ ! -f "./vocabulary/ORBvoc.bin" ];then
  if [ ! -f "./vocabulary/ORBvoc.txt" ];then
    cd vocabulary
    tar -xf ORBvoc.txt.tar.gz
    cd ..
  fi
  ./tools/bin_vocabulary ./vocabulary/ORBvoc.txt ./vocabulary/ORBvoc.bin
  rm vocabulary/ORBvoc.txt
fi
echo "Finish !"
