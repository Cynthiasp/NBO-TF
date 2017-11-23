from textblob import TextBlob

wiki = TextBlob( 'Together, we are going to restore safety to our streets and peace to our communities, and we are going to destroy the vile criminal cartel, #MS13, and many other gangs...')
wiki.tags
wiki.words

#sentiment
x = wiki.sentiment.polarity

print x
