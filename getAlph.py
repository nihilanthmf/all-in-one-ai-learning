f = open("words_alpha.txt", "r")

s = f.read()

f.close()

alph = set()
for i in s:
    alph.add(i)

print("".join(alph))