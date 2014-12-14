rm -f NNdemo.profile
rm -f callgrindNNdemo.log

valgrind  --tool=callgrind --dump-before=dlclose  --callgrind-out-file=NNdemo.profile ./NNdemo  #| tee callgrindNNdemo.log

callgrind_annotate --inclusive=yes --tree=both --auto=yes NNdemo.profile 
echo "to see the profile do \"kcachegrind NNdemo.profile\""
kcachegrind NNdemo.profile


