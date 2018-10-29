# raws
Real-time Automatic Word Segmentation (for user-generated texts)

## Word Vector 
[Pretrained 100dim fastText vector for Korean](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)
* Download this and unzip THE FOLDER in the same folder with 'kor_spacing.py' 
* Loading the model will be processed by load_model('vectors/model')
* This model is identical to [Ttuyssubot](https://github.com/warnikchow/ttuyssubot)

[GloVe Twitter 27B for English](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
* Download this and locate 100dim dictionary to the same folder with 'hashseg.py', in file name 'glove100.txt'.
* Dictionary-free version is under implementation!
* This model is identical to [fdhs](https://github.com/warnikchow/fdhs)

## System description
