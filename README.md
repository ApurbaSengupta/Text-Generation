# Text-Generation

Developing a LSTM-based deep neural language model to generate text sequences character-by-character, given a context sequence of text.

Utilizing few English novels for training and evaluating the model.

Utilizing PyTorch framework for development. Using a NVIDIA GeForce GTX 1080 Ti GPU machine to facilitate training of the model.


## Text generation examples

**From training novels**:

**1. Pride and Prejudice - Jane Austen**:

...**Context** : The tumult of her mind, was now painfully great. She knew not how to support herself, and from actual weakness sat down andcried for half-an-hour. 

...**Generated**: *At elobor of a day--but the elmost Farkance were the grounds the day, let the part-accation, and her knees, and she had felt the other fellow for the ofference of some days which she sook in once down to the rising feeling dinner she in the water from most once enough to your deay so much on the hor*

**2. Dracula - Bram Stoker**:

...**Context** : To believe in things that you cannot. Let me illustrate. I heard once of an American who so defined faith: 'that faculty which enables us to believe things which we know to be untrue.' For one, I follow that man. 

...**Generated**: *The rest each other point-hot of the earth, let she had defered the sentiment cask and to be prisoner enough to chood for it. He saw the forecast rew viious as a fair and of the past of the friend and carising now often the days are to erlive the more each of his life. He had inleight. "With my shoc*


**From non-training novels (but same author or same genre)**:

**1. Emma - Jane Austen**:

...**Context** : In short, she sat, during the first visit, looking at Jane Fairfax with twofold complacency; the sense of pleasure and the sense of rendering justice, and was determining that she would dislike her no longer. 

...**Generated**: *"And you might _pity, I do not see me all. It was dribured. I will only have forting Monseigneur in his good and conversation in a neither aloft in a known."

"_Younse between what a horrible more cowfort of the carriage, burne, so I longed me many through a dream. For you
shall I told them in a sho*

**2. The Strange Case of Dr. Jekyll and Mr. Hyde - Robert Louis Stevenson**:

...**Context** : Poole swung the axe over his shoulder; the blow shook the building, and the red baize door leaped against the lock and hinges. A dismal screech, as of mere animal terror, rang from the cabinet. 

...**Generated**: *The pain so-finding to the offered to say him so well with a punison. It was gone the catter say the composed as the deck. The possible, and in given the fellow for so locked in without bone of his form for it, and one, the world of rocks of a great procious ladyshipon their life, in Concernse of e*
