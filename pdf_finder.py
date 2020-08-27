import numpy as np

def pdf_finder(coordinates,ratio1=1.2,ratio2=2.1):
	if len(coordinates)!=6:
		print('Molim  uneseti brojeve koje ispunjavaju kordinate (0,n1),(0,n2),(n3,0),(n4,n5),(n6,0) tim redom')
		return -1
	else:
		# trazi oblasti povrsina
		area1 = (2*coordinates[2]) * (2*coordinates[1])
		area2 = (2*coordinates[3]) * (2*coordinates[0]) - area1
		area3 = 2*(coordinates[5]-coordinates[3]) * (2*coordinates[4])

		# trazi verovatnoce
		prob = np.ones((3,1))
		prob[0] = 1/(area1+ratio1*area2+ratio2*area3)
		prob[1] = ratio1*prob[0];
		prob[2] = ratio2*prob[1];
		return prob



