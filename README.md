# raws
Real-time Automatic Word Segmentation (for user-generated texts)

## Word Vector 
Korean: [Pretrained 100dim fastText vector](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)
* Download this and unzip THE FOLDER in the same folder with 'kor_spacing.py' 
* Loading the model will be processed by load_model('vectors/model')
* This model is identical to [Ttuyssubot](https://github.com/warnikchow/ttuyssubot)

English: [GloVe Twitter 27B pretrained](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
* Download this and locate 100dim dictionary to the same folder with 'hashseg.py', in file name 'glove100.txt'.
* Dictionary-free version is under implementation!
* This model is identical to [fdhs](https://github.com/warnikchow/fdhs)

## System description
* Korean spacing<br/>
<image src="https://github.com/warnikchow/raws/blob/master/fig2.png" width="700"><br/>
* English hashtag segmentation<br/>
<image src="https://github.com/warnikchow/raws/blob/master/fig2_5.png" width="700"><br/>
* Specification<br/>
<image src="https://github.com/warnikchow/raws/blob/master/table.PNG" width="500"><br/>
