x = 6800000
y = 0
for i in range(10):
    y += x * 0.017 / 2
    print(x, y, y / 60)
    x += 6800000