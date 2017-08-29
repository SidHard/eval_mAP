import matplotlib.pyplot as plt
import eval

rec, prec, ap = eval.eval('val.txt', 'result.txt')

plt.plot(rec, prec, 'r-')
plt.title('ap: ' + str(ap))
plt.xlabel('recall')
plt.ylabel('precision')
plt.savefig('pr.jpg')
plt.show()