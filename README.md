# Text-Generation

An attention-based deep neural language model using biidrectional LSTMs to generate text sequences character-by-character, given a context sequence of text.

Utilized few English novels for training and evaluating the model.

Utilized PyTorch framework for development. Used a NVIDIA GeForce GTX 1080 Ti GPU machine to facilitate training of the model.

The reduction in training loss against number of epochs the model is trained for is shown below. The model achieves an average validation perplexity of approximately 67.

The trained model is available as "model_generate.pt".


## Text generation examples

### From training novels:

**1. Pride and Prejudice - Jane Austen**:

...**Context** : The tumult of her mind, was now painfully great. She knew not how to support herself, and from actual weakness sat down andcried for half-an-hour. 

...**Generated**: *As I found that I think my cold--ship of any command, obstructive, and in the closely. What was all good scarcely distress,when you can the rise and then immen that she seek sound him that you are more night be. Jane so friend! There was of the steps of a tran by the sea all the room for her. When*

**2. Dracula - Bram Stoker**:

...**Context** : To believe in things that you cannot. Let me illustrate. I heard once of an American who so defined faith: 'that faculty which enables us to believe things which we know to be untrue.' For one, I follow that man. 

...**Generated**: *Go not things, he might I be things to any man where any dark to call a harders is from the matter carriage, not only starm sunset from the day.*

*"I don't want of entire more so receiving the spirit in the room was a great us and such a mist we should go it, and the window glanced from you get to re*


### From non-training novels (but same author or same genre):

**1. Emma - Jane Austen**:

...**Context** : During his present short stay, Emma had barely seen him; but just enough to feel that the first meeting was over, and to give her the impression of his not being improved by the mixture of pique and pretension, now spread over his air. 

...**Generated**: *fish, for I can don, that the sat up in the passions at all, for my soon with exoloce, for my mark who had not to be that he had long considerable the smalled by the sobubs; when the degree was, authorded you at land with the more gloped on*

**2. The Strange Case of Dr. Jekyll and Mr. Hyde - Robert Louis Stevenson**:

...**Context** : Poole swung the axe over his shoulder; the blow shook the building, and the red baize door leaped against the lock and hinges. A dismal screech, as of mere animal terror, rang from the cabinet. 

...**Generated**: *But all the rest on the London. When the shapeded, that strength of the bell of the druck. "_Court_, also insported me, "that they would find there well, and then that time lad hepromided that my back, the ground, who sun I would never us. It is a secrets from this straight that both suspicion to t*
