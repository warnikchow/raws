# raws
Real-time Automatic Word Segmentation (for user-generated texts)

## Requirements
fastText, Keras (TensorFlow), Numpy

## Word Vector 
Korean: [Pretrained 100dim fastText vector](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)
* Download this and unzip in NEW FOLDER 'vector' in the name 'model_kor.bin'. 
* Loading the model will be processed by fasttext.load_model('vectors/model_kor')
* This model is identical to [Ttuyssubot](https://github.com/warnikchow/ttuyssubot)

English: [GloVe Twitter 27B pretrained](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
* Download this and locate 100dim dictionary (.txt) to NEW FOLDER 'vector' in the name 'momdel_eng.txt'.
* Loading the model will be processed by loadvector('vectors/model_eng')
* Dictionary-free version is under implementation!
* This model is identical to [fdhs](https://github.com/warnikchow/fdhs)

## How to use
* Easy start: Python3 execute file
<pre><code> python3 raws_demo.py </code></pre>
* This system assigns a contextual spacing for conversation-style and non-normalized text
- ex1) 아버지친구분당선되셨더라 >> "아버지 친구분 당선 되셨더라"
- ex2) 너본지꽤된듯 >> "너 본지 꽤 된 듯"
- ex3) Mamorizatcambrige >> "Mamoriz at cambrige"
* For Korean, the spacing may not be strictly correct, but the system was trained in a way to give a plausible duration for speech synthesis, in the aspect of a non-canonical spoken language.
* For English, the segmentation can be inaccurate for the literary texts, but robust to the errata or user-generated noisy texts.

### Importing automatic spacer for Korean
<pre><code> from raws import kor_spacing as spc </code></pre>
* Sample usage:
<pre><code> spc('나얼만큼사랑해') 
 >> '나 얼만큼 사랑해' </code></pre>
<pre><code> spc('아버지친구분당선되셨더라')  
 >> '아버지 친구분 당선 되셨더라' </code></pre>
 <pre><code> spc('역사를하노라고땅을파다가커다란돌을하나끄집어내어놓고보니도무지어디서인가본듯한생각이들게모양이생겼는데목도들이그것을메고나가더니어디다갖다버리고온모양이길래쫓아나가보니위험하기짝이없는큰길가더라그날밤에한소나기하였으니필시그돌이깨끗이씻꼈을터인데그이틀날가보니까변괴로다간데온데없더라어떤돌이와서그돌을업어갔을까나는참이런처량한생각에서아래와같은작문을지었다내가그다지사랑하던그대여내한평생에차마그대를잊을수없소이다내차례에못을사랑인줄은알면서도나혼자는꾸준히생각하리라자그러면내내어여쁘소서어떤돌이내얼굴을물끄러미치어다보는것만같아서이런시는그만찢어버리고싶더라')  
 >> '역사를 하노라고 땅을 파다가 커다란 돌을 하나 끄집어 내어놓고 보니 도무지 어디서 인가본 듯한 생각이 들게 모양이 생겼는데 목도들이 그것을 메고 나가더니 어디다 갖다 버리고 온 모양이길래 쫓아 나가보니 위험하기 짝이없는 큰 길 가더라 그날밤에 한소나기하였으니 필시그 돌이 깨끗이 씻꼈을 터인데 그 이틀 날가보니까 변괴로 다간데 온데 없더라 어떤 돌이 와서 그 돌을 업어갔을까 나는 참 이런 처량한 생각에서 아래와 같은 작문을 지었다 내가 그다지 사랑하던 그대여 내한 평생에 차마 그대를 잊을수 없소이다 내 차례에 못을 사랑인줄은 알면서도 나 혼자는 꾸준히 생각하리라 자 그러면 내내어여쁘소서 어떤 돌이 내 얼굴을 물끄러 미치어 다 보는 것만 같아서 이런 시는 그만 찢어버리고 싶더라' </code></pre>
### Importing hashtag segmentation toolkit for English:
<pre><code> from raws import eng_hashsegment as hashseg </code></pre>
<pre><code> from raws import eng_segment as seg </code></pre>
* Sample usage:
<pre><code> hashseg('#what_do_you_want') 
 >> 'what do you want' </code></pre>
<pre><code> hashseg('#WhatDoYouWant')  
 >> 'What Do You Want' </code></pre>
<pre><code> hashseg('#whatdoyouwant') 
 >> 'what do you want' </code></pre>
<pre><code> seg('whatdoyouwant') 
 >> 'what do you want' </code></pre>
<pre><code> seg('therearesomanynonsegmentedtextsontheearthandweareaimingtodealwiththose') 
 >> 'there are so many non segmented texts on the earth and we are aiming to deal with those' </code></pre>
 
## System description
### Korean spacing<br/>
<image src="https://github.com/warnikchow/raws/blob/master/images/fig2.png" width="700"><br/>
### English hashtag segmentation<br/>
<image src="https://github.com/warnikchow/raws/blob/master/images/fig2_5.png" width="700"><br/>
### Specification<br/>
<image src="https://github.com/warnikchow/raws/blob/master/images/table%60.PNG" width="500"><br/>
 
## Citation 
### As a Korean word vector dictionary, toolkit, and a concept of spacer targetting noisy user-generated text
```
@article{cho2018real,
	title={Real-time Automatic Word Segmentation for User-generated Text},
	author={Cho, Won Ik and Cheon, Sung Jun and Kang, Woo Hyun and Kim, Ji Won and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1810.13113},
	year={2018}
}
```
 
### DISCLAIMER: This model is trained with drama scripts and targets user-generated noisy texts; for the accurate spacing of literary style texts, refer to [PyKoSpacing](https://github.com/haven-jeon/PyKoSpacing)

## Demonstration (for Korean)
* https://www.youtube.com/watch?v=mcPZVpKCH94&feature=youtu.be
