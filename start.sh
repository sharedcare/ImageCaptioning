wget -nv https://s3.us-east-2.amazonaws.com/iltc-public/flickr8k-new.zip
unzip -q flickr8k-new.zip -d flickr8k
python generator_22.py

cp checkpoint22.h5 /output
cp model22.h5 /output
cp -r logs /output