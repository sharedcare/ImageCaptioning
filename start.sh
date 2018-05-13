wget -nv https://s3.us-east-2.amazonaws.com/iltc-public/flickr8k-new.zip
unzip -q flickr8k-new.zip -d flickr8k
python train.py

cp checkpoint.h5 /output
cp model.h5 /output
cp -r logs /output