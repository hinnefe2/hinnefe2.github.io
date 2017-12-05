---
layout: post
title: "An alethiometer for the modern age"
tags:
    - python
    - tools
categories: 
    - python 
    - tools
--- 

*The Golden Compass* was one of my favorite books growing up. It has lots of
your standard young adult fantasy epic elements -- a plucky heroine, talking
animals, authoritarian villians -- but it also touches on some weighty
theological themes. The author described it as a deliberate inversion of
Milton's *Paradise Lost* (and not for nothing, at the end of the series the
protagonists save the world by killing God and re-committing original sin).  A
central element in the book is the existence of the eponymous "golden compass",
a literal *machina qua deus ex* which answers questions through divine
intervention. The compass presents its answers as a series of ideograms: its
face is ringed with symbols and when posed a question its needle sweeps around
the face selecting the symbols which comprise the answer.  I always wanted one
of those when I was a kid but, alas, back then powerful artifacts with oracular
capabilities were in short supply. Nowadays we have smartphones and twitter
though so better late than never! In this post I'm going to describe a twitter
bot I made which answers questions with emoji (hence
[*alethiomoji*](https://twitter.com/alethiomoji), the name of the project; the
golden compass was also called an alethiometer).

This is what it looks like in action: 
 
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr"><a
href="https://twitter.com/alethiomoji">@alethiomoji</a> is this the end of the
the world?</p>&mdash; Henry Hinnefeld (@DrJSomeday) <a
href="https://twitter.com/DrJSomeday/status/824076525817491456">January 25,
2017</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet" data-lang="en"><p lang="und" dir="ltr"><a
href="https://twitter.com/DrJSomeday">@DrJSomeday</a> 🔚 🌐 ⏳</p>&mdash; Emoji
Golden Compass (@alethiomoji) <a
href="https://twitter.com/alethiomoji/status/824076719292317698">January 25,
2017</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script> 
 
In the book interpreting the Compass is not straightforward; it takes some
creativity to pick out the right meaning for each symbol. For example, the
kettle symbol could mean 'food' but it could also mean 'plan' because cooking
follows recipes which are like plans. This gives us some leeway in making our
emoji version: as long as we can come up with emoji that are somewhat related
to the words in a given question we can rely on people's imagination to fill
in the gaps.

The general plan then is to:

1. Pick out semantically important words from a given question.
2. Find emoji which are related to each of the important words.
3. Wrap things up in some machinery to read from and post to twitter.

Note that bot doesn't actually try to 'answer' the question in any meaningful
way: under the hood it's just finding emoji which are related to the important
words in the question. I also made each response include an extra emoji that
can be interpreted as a yes / no / maybe so that the responses feel more like
answers. The code is on github
[here](https://github.com/hinnefe2/alethiomoji); in this post I'll sketch out
how the interesting bits work. I used existing python modules for parts 1 and
3 so the focus will be mostly on part 2. 
 
## Finding semantically important words

To find the semantically important words in each question I ran the question
text through [stat_parser](https://github.com/emilmont/pyStatParser). This
produces a parsed sentence tree for the question and labels each word with a
part of speech tag. Parsing the question this way does limit the questions
Alethiomoji can answer to those which `stat_parser` can parse, however in
practice this doesn't seem to be a big limitation. I chose nouns, verbs, and
adjectives as the semantically interesting words, so once the question is
parsed we use the part of speech tags to pull out the relevant words and pass
them on to the next step. 
 
## Matching words to emoji with tf-idf

Once we have the semantically important words we need to somehow match them to
emoji. One place to start is with the [official
descriptions](http://unicode.org/emoji/charts/full-emoji-list.html) of each
emoji. Conveniently for us, the folks at
[emojityper.com](https://emojityper.com/) have already scraped all the
descriptions into a nice, tidy csv
[file](https://github.com/emojityper/emojityp
er.github.io/blob/master/res/emoji/annotations.txt).

We can use Scikit-learn's [CountVectorizor](http://scikit-
learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage) to
vectorize each of the emoji descriptions. This gives us an $N_\text{emoji}$ by
$N_\text{words}$ matrix, where each column is associated with one word (among
all the words that show up in the descriptions) and each row is associated with
an emoji. To avoid giving too much emphasis to common words we can run this
matrix through scikit-learn's [tf-
idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) transform to weight
different words by how common or uncommon they are. 


{% highlight python %}
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
{% endhighlight %}


{% highlight python %}
annot_df = pd.read_csv('alethiomoji/annotations.txt')

vectorizer = CountVectorizer().fit(annot_df.annotation)
count_matrix = vectorizer.transform(annot_df.annotation)

transformer = TfidfTransformer()
count_tfidf = transformer.fit_transform(count_matrix)
{% endhighlight %}
 
Now that we have this matrix we can find emoji related to any word that shows
up in the emoji descriptions. To do this we run the input word through the same
`CountVectorizor` then multiply the resulting vector by the tf-idf matrix to
get the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
between that word and each emoji.

For example, running this process against the word 'dog' gives: 


{% highlight python %}
word = 'dog'

# word must be in a list, otherwise vectorizer splits it into characters instead of words
word_vector = vectorizer.transform([word])

# calculate cosine similarity of the word vector with each emoji's annotation vector
cos_sim = count_tfidf.dot(word_vector.transpose())
{% endhighlight %}


{% highlight python %}
(pd.DataFrame(data = cos_sim.toarray().flatten(),
              index = annot_df.unicode,
              columns = [word])
   .sort_values(by=word, ascending=False)
   .head(5))
{% endhighlight %}




<div>
<table border="1" class="dataframe center">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dog</th>
    </tr>
    <tr>
      <th>unicode</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>🐕</th>
      <td>0.646669</td>
    </tr>
    <tr>
      <th>🐶</th>
      <td>0.612562</td>
    </tr>
    <tr>
      <th>🐩</th>
      <td>0.596183</td>
    </tr>
    <tr>
      <th>🌭</th>
      <td>0.393474</td>
    </tr>
    <tr>
      <th>😀</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


 
This works well for words which show up in the emoji descriptions, but that's a
small subset of the words we might encounter (there are only about 1500
distinct words in the descriptions). To expand our vocabulary we need to come
up with another way to compare the similarity of emoji and words. Fortunately
someone else has done that for us. 
 
## Matching words to emoji with word2vec 
 
Some researchers at University College London and Princeton took the same emoji
descriptions we used above, along with a manually curated set of annotations
and ran them all through the Google News word2vec
[model](https://code.google.com/archive/p/word2vec/). Their
[paper](https://arxiv.org/pdf/1609.08359.pdf) has more details about their
methodology, but for our purposes the main result is that they
[released](https://github.com/uclmr/emoji2vec) a set of word2vec vectors for
emoji.

Using these emoji word2vec vectors and the original Google News model we can do
the same thing we did above: start with a word, get its vector, multiply that
vector by the matrix of emoji vectors, and then check the resulting cosine
similarities.

For example, running this on the word 'apocalypse' gives: 


{% highlight python %}
import sqlite3
{% endhighlight %}


{% highlight python %}
word = 'apocalypse'

# the word2vec vectors are stored in a local sqlite database
db_conn = sqlite3.connect('alethio.sqlite')

# the vectors for emoji are in a table called 'emoji_w2v'
emoji_vecs = (pd.read_sql('SELECT * FROM emoji_w2v', db_conn, index_col='unicode')
                .apply(pd.to_numeric))

# the vectors for words are in a table called 'words_w2v'
word_vector = (pd.read_sql("SELECT * FROM words_w2v WHERE word='{}'".format(word), 
                           db_conn, index_col='word')
                 .apply(pd.to_numeric))
{% endhighlight %}


{% highlight python %}
emoji_vecs.dot(word_vector.transpose()).sort_values(by=word, ascending=False).head(5)
{% endhighlight %}


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>word</th>
      <th>apocalypse</th>
    </tr>
    <tr>
      <th>unicode</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>👹</th>
      <td>0.893812</td>
    </tr>
    <tr>
      <th>🔪</th>
      <td>0.836191</td>
    </tr>
    <tr>
      <th>👾</th>
      <td>0.802000</td>
    </tr>
    <tr>
      <th>💀</th>
      <td>0.750207</td>
    </tr>
    <tr>
      <th>😵</th>
      <td>0.723631</td>
    </tr>
  </tbody>
</table>
</div>

 
It's worth noting that we could just use the word2vec approach and not bother
with the annotations and tf-idf, but one perk of using the annotations is that
we can add in custom associations between emoji and certain words. 
 
## Wrapping things up 
 
With that we're pretty much done. All that's left is to use the cosine
similarities to choose an emoji for each word and then connect everything to
twitter. For the first part we can use the similarities to weight a random
selection with `numpy.random.choice`, and for the second part we can use the
([twython](https://twython.readthedocs.io/en/latest/)) library to communicate
with the twitter API. Head on over the the github
[repo](https://github.com/hinnefe2/alethiomoji) to see how that all works.

The last thing to sort out is where to actually run the bot. For simplicity's
sake I ended up running everything on an AWS, in a t2.nano instance. These
instances only have 500MB of RAM which isn't enough to hold all the word2vec
vectors in memory, but querying a local sqlite database is plenty fast for our
purposes. This could probably be an AWS Lambda function too but we'll save that
for version 2.

And that's all there is too it, head on over to
[@alethiomoji](https://twitter.com/alethiomoji) and check it out. 
