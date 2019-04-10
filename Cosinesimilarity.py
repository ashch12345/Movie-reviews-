#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(binary=True) #explore other parameters to
corpus=['''Summer is a charming flirt. Easygoing and casual. summer doesn't huff and puff to win our affections. It has us at "Hello." Winter broods like the tortured protagonist of big fat Russian novel. It is daunting and dramatic, burning with a slow intensity.
The season's reputation precedes itslef, and often, not in a good way. It has a way of whittling down everything to its bare bones. Even relationships not attuned to its ebbs and flows can fray. At a dinner conversation I once attended, I listened in bemusement as a recent divorcee made the case that it was the Scandinavian frost that had cooled his ex-wife's ardour. How original.


One of the the finer books I read this year was John Kaags Hiking with Nietzsche, in which Kaag, a professor of philosophy, rekindles his passion for the German thinker while tracing picturesque iking trails in the mountains of Switzerland. 
It's a near precise rendering of the travelogue as a self help book. A young Kaag was an avowed Nietzsche acolyte but given the ravages of responsibilities and adulthood, the writer put his affinity to test by undertaking physically enduring hikes through the Alps, revisiting haunts that the philosopher escaped to, in search of solitude and salve. The journey's demands, coupled with his own inner turmoil, are catnip for anybody feeling at cross purposes with their own life.
In the book, Kaag qyites Neitzsche writing to his mother after he had spent time in Splugen, "I was overcome by the desire to remain here... this high alpine valley... There are pure, strong gusts of air, hills and boulders of all shapes... But what pleases me the most are the splendid highroads over which I walk for hours." Travel as the answer to searching questions is harddly a radical idea but what's endearing about the book is that it subtly confirms a basic tenet of why we go on these journeys in the first place. Sometimes, being on the move matters more than anything else.''','''


Winter travel is an exercise in negotiation, seepecially for sunshine souls. "How many extra clothes do I have to pack now?""The weather is minus what-did-you-say?" All valid concerns but the recommendations far outweigh them. Take one trivial scorng point: the winter wardrobe, which is tred chic, and can make the most sartorially challenged among us look like runway models.
The allure of winter lies in nature -- so immense, overwhelming and, of course, achingly beautiful. In his collection of letters to an unborn daughter, Norwegian Karl Ove Knausgard meditates on the sounds of the season.''']


# In[2]:


vect.fit(corpus)


# In[6]:


print(vect.transform(["Today is good optical"]).toarray())


# In[7]:


print(vect.transform(corpus).toarray())


# In[5]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vect.transform(corpus).toarray())
print(similarity)


# In[ ]:




