# Continued-PreTraining
Repository to test the continued pre-training of a foundation model with domain specific data, in this case Irish language and Dáil debates.

## Early Results
Here is its response when asked to tell a story about Irish politics. First the original model, then the model with a little bit of training. After training, the model starts saying some real Irish words and structuring sentences. While the model remains nonsensical, it is encouraging to see some improvement with such a tiny model and small subset of data. 

Question:

Inis dom scéal faoi pholaitíocht na hÉireann. 

Original:

Hhe fheir fheir na hÉireann na hheir hheir fheir na hÉireann hheir hheir fheir hheir na hÉireann fheir fheir hheir na hÉireann. Hheir fheir hheir hheir fheir fheir hheir na hÉireann. Na hheir hheir hheir fheir fheir fheir fheir hheir hheir na hÉireann. Na hheir hheir fheir hheir fheir hheir fheir hheir fheir hheir hheir na hÉireann.

Okay, so I need to figure out what this sentence is about. The user wrote in Irish, so

Trained: 

2. Tá na Páirtithe Eireann, tá sé a bhfuil aon idirdhealú agus aon idirdhealú na hEorání Chomhchoiste iomailte agus iompair a bhfuil aon sin na preocétaí don chomhchoiste sárú, mar a bhfuil na cuspóiríocht de na forálacha iomaíla atá atáirgaidh íocú, deagú iomgáit agus is féidir na bearta is gá a ghabhú, na bearta beátaí a leaghaíocht don Chomhchoiste Ma
