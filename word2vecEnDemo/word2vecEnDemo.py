import word2vec
"""
word2vec.word2phrase('text8', 'text8-phrases', verbose=True)
word2vec.word2vec('text8-phrases', 'text8.bin', size=100, verbose=True)
word2vec.word2clusters('text8', 'text8-clusters.txt', 100, verbose=True)

"""
print("-------------word2vec example-----------")
model = word2vec.load('text8.bin')
print("-------------ankara similar---------------")
indexes, metrics = model.cosine('ankara') 
print(model.generate_response(indexes, metrics).tolist()[:5])
print("----------los_angeles similar-------")
indexes, metrics = model.cosine('los_angeles')
print(model.generate_response(indexes, metrics).tolist()[:5])
print("__________king+woman-man____________")
indexes, metrics = model.analogy(pos=['king', 'woman'], neg=['man'], n=5)
print(model.generate_response(indexes, metrics).tolist())
print("---------------------------------")