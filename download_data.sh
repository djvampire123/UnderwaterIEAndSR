mkdir data
gdown --fuzzy -O data/data.zip https://drive.google.com/file/d/1vRl6IIX7YbubCyA7ju_7Yl8S27gB9OpV/view
cd data && unzip data.zip && rm data.zip
cp -r /content/drive/MyDrive/Underwater/UnderwaterIEAndSR/data/content/drive/MyDrive/Underwater/UnderwaterIEAndSR/data/* /content/drive/MyDrive/Underwater/UnderwaterIEAndSR/data
rm -r /content/drive/MyDrive/Underwater/UnderwaterIEAndSR/data/content