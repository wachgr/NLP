word = 'dearz'
splits_a=[]
for w in range(len(word)+1) :
    splits_a.append([word[:w],word[w:]])

print(splits_a)
#delete edit
splits = splits_a
deletes = []
print('word:',word)
for l,r in splits:
    if r:
        print(word,"delete",r[0],'-->',l+r[1:])
#
# word = 'hengyanyuchaoyouxiude'
# for i in range(len(word)):
#     print(word[i])

splits = splits_a
deletes = [L + R[1:] for L, R in splits if R]
print(deletes)
vocab = ['dean','deer','dear','fries','and','coke']
edits = list(deletes)
print('vocab : ', vocab)
print('edits : ', edits)
candidates=[]