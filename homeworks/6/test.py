import random

ws = ''
with open('/usr/share/dict/words') as f:
    for line in f:
        if line.startswith('w') and random.random() < 0.75:
            word = 'W' + line[1:-1] + ' '
            ws = ws + word
            print(ws)

print('Weave\n')
