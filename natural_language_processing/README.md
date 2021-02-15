# Natural Language Processing 

Natural Language Processing (or NLP for short) is a discipline in computing that deals with the communication between natural (human) languages and computer languages. A common example of NLP is something like spellcheck or autocomplete. Essentially NLP is the field that focuses on how computers can understand and/or process natural/human languages. 

### Recurrent Neural Networks

We will learn how to use a reccurent neural network to do the following:
- Sentiment Analysis
- Character Generation 

RNN's are complex and come in many different forms so in this tutorial we wil focus on how they work and the kind of problems they are best suited for.

## Sequence Data

Previously, we focused on data that we could represent as one static data point where the notion of time or step was irrelevant. Take for example our image data, it was simply a tensor of shape (width, height, channels). That data doesn't change or care about the notion of time. 

We will look at sequences of text and learn how we can encode them in a meaningful way. Unlike images, sequence data such as long chains of text, weather patterns, videos and really anything where the notion of a step or time is relevant needs to be processed and handled in a special way. 

But what do I mean by sequences and why is text data a sequence? Well that's a good question. Since textual data contains many words that follow in a very specific and meaningful order, we need to be able to keep track of each word and when it occurs in the data. Simply encoding say an entire paragraph of text into one data point wouldn't give us a very meaningful picture of the data and would be very difficult to do anything with. This is why we treat text as a sequence and process one word at a time. We will keep track of where each of these words appear and use that information to try to understand the meaning of peices of text.

## Encoding Text


***Refer to the Text Encoding Noteboook***

