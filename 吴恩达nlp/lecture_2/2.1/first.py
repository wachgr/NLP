import re
from collections import Counter
import matplotlib.pyplot as plt
text = 'red pink pink blue blue yellow ORANGE BLUE BLUE PINK'
print(len(text))
text_lowercase = text.lower()
print(text_lowercase)
#将字符串转化成数组，\w+匹配一个或多个连续的字母数字字符（包括下划线），即一个单词
#re模块就用于在字符串中找到匹配的项
words = re.findall(r'\w+',text_lowercase)
print(words)
vocab = set(words)
print(vocab)
count_a = dict()
for w in words:
    count_a[w] = count_a.get(w,0) + 1
print(count_a)
#可迭代对象（如列表、元组、字符串等）传递给Counter的构造函数，它会计算每个元素的出现次数。
counts_b = Counter(words)
print(counts_b)

d = {'blue': counts_b['blue'], 'pink': counts_b['pink'], 'red': counts_b['red'], 'yellow': counts_b['yellow'], 'orange': counts_b['orange']}
plt.bar(range(len(d)),list(d.values()),align='center',color=d.keys())
# _=plt.xticks(range(len(d)),list(d.keys()))
plt.show()

plt.bar(list(d.keys()),list(d.values()),align='center',color=d.keys())
plt.show()