import matplotlib.pyplot as plt


months= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
pass_2015 = [0, 0, 0, 0, 0, 134, 115, 185, 179, 147, 160, 126, 208, 48, 47, 50]
pass_2017 = [34, 106, 98, 162, 128, 166, 117, 123, 225, 161, 110, 100, 144, 53, 27, 10]
pass_2019 = [141,175,166,227,302,302,314]
plt.plot(months, pass_2015, color='gray')
plt.plot(months, pass_2017, color='green')
plt.plot(months[:len(pass_2019)], pass_2019, color='blue')
plt.show()
