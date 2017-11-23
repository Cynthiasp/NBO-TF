{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-0.05625\n"
     ]
    }
   ],
   "source": [
    "from textblob import TextBlob\n",
    "\n",
    "wiki = TextBlob( 'Together, we are going to restore safety to our streets and peace to our communities, and we are going to destroy the vile criminal cartel, #MS13, and many other gangs...')\n",
    "wiki.tags\n",
    "wiki.words\n",
    "\n",
    "#sentiment\n",
    "x = wiki.sentiment.polarity\n",
    "\n",
    "print x\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
