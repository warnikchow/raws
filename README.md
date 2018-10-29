# raws
Real-time Automatic Word Segmentation (for user-generated texts)

## Word Vector 
Korean: [Pretrained 100dim fastText vector](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)
* Download this and unzip THE FOLDER in the same folder with 'raws.py' 
* Loading the model will be processed by load_model('vectors/model')
* This model is identical to [Ttuyssubot](https://github.com/warnikchow/ttuyssubot)

English: [GloVe Twitter 27B pretrained](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
* Download this and locate 100dim dictionary to the same folder with 'hashseg.py', in file name 'glove100.txt'.
* Dictionary-free version is under implementation!
* This model is identical to [fdhs](https://github.com/warnikchow/fdhs)

## How to use
* Easy start: Python3 execute file
<pre><code> python3 raws_demo.py </code></pre>
* This system assigns a contextual spacing for conversation-style and non-normalized Korean text
- ex1) 아버지친구분당선되셨더라 >> "아버지 친구분 당선 되셨더라"
- ex2) 너본지꽤된듯 >> "너 본지 꽤 된 듯"
- ex3) 뭣이중헌지도모름서 >> "뭣이 중헌지도 모름서"
- ex4) 나얼만큼사랑해 >> "나 얼만큼 사랑해"
* The spacing may not be strictly correct, but the system was trained in a way to give a plausible duration for speech synthesis, in the aspect of a non-canonical spoken language.
### Importing automatic spacer for Korean
<pre><code> from raws import kor_spacing as spc </code></pre>
* Sample usage:
<pre><code> spc('나얼만큼사랑해') 
 >> '나 얼만큼 사랑해' </code></pre>
<pre><code> spc('아버지친구분당선되셨더라')  
 >> '아버지 친구분 당선 되셨더라' </code></pre>
### Importing hashtag segmentation toolkit for English:
<pre><code> from raws import eng_segment as seg </code></pre>
* Sample usage:
<pre><code> seg('#what_do_you_want') 
 >> 'what do you want' </code></pre>
<pre><code> seg('#WhatDoYouWant')  
 >> 'What Do You Want' </code></pre>
<pre><code> seg('#whatdoyouwant') 
 >> 'what do you want' </code></pre>

## System description
### Korean spacing<br/>
<image src="https://github.com/warnikchow/raws/blob/master/fig2.png" width="700"><br/>
### English hashtag segmentation<br/>
<image src="https://github.com/warnikchow/raws/blob/master/fig2_5.png" width="700"><br/>
### Specification<br/>
<image src="https://github.com/warnikchow/raws/blob/master/table.PNG" width="500"><br/>

### DISCLAIMER: This model is trained with drama scripts and targets user-generated noisy texts; for the accurate spacing of literary style texts, refer to [PyKoSpacing](https://github.com/haven-jeon/PyKoSpacing)

## Demonstration (for Korean)
* https://www.youtube.com/watch?v=mcPZVpKCH94&feature=youtu.be
