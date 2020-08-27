import numpy as np
from pdf_finder import pdf_finder
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
from numpy import linalg as LA

def find_prob(coordinates,test_points,probabilities):	# trazi gustinu verovatnoce u prostoru
	prob_points = []
	for index,[x,y] in enumerate(test_points):
		if (x<coordinates[2] and x>-coordinates[2] and y<coordinates[1] and y>-coordinates[1]):
			prob_points.append(probabilities[0])
		elif (x<coordinates[3] and x>-coordinates[3] and y<coordinates[0] and y>-coordinates[0]):
			prob_points.append(probabilities[1])
		else:
			prob_points.append(probabilities[2])	
	return prob_points

def initial_estimate(coordinates,region_number):	# nasumicne pocetne procene tezista kvantizacije
	ax = coordinates[5] * (2*np.random.rand(region_number,1)-1) # inicijalna procena ax izmedju -n6 and n6
	ay = coordinates[0] * (2*np.random.rand(region_number,1)-1) # initial estimates of ax izmedju -n1 and n1
	a = np.concatenate((ax,ay),axis=1)
	return a

def generate_points(coordinates):	# vraca samplovanu verziju prostora
	test_points = []
	for x in [float(j)/10 for j in range(-coordinates[5]*10,coordinates[5]*10)]:
		for y in [float(h)/10 for h in range(-coordinates[4]*10,coordinates[4]*10)]:
			test_points.append([x,y])
	for x in [float(j)/10 for j in range(-coordinates[3]*10,coordinates[3]*10)]:
		for y in [float(h)/10 for h in range(coordinates[4]*10,coordinates[0]*10)]:
			test_points.append([x,y])
		for y in [float(h)/10 for h in range(-coordinates[0]*10,coordinates[4]*10)]:
			test_points.append([x,y])
	return test_points


coordinates = []
print('Unesite brojeve n1 do n6   sajednickog pdf-a. Format kordinata je (0,n1),(0,n2),(n3,0),(n4,n5),(n6,0) ovim redosledom.')
for i in range(6):
	n = input('Unesi broj: ')
	coordinates.append(int(n))
print('Brojevi :',coordinates)

ratio1 = float(input('Unesi kolicnik odnosa p2 i p1 '))
ratio2 = float(input('Unesi kolicnik odnosa p3 i p1 '))
probabilities = pdf_finder(coordinates,ratio1,ratio2)

print('Gustine verovatnoce funkcija na povrsini su ',probabilities)

region_number = int(input('Unesi broj  diskretizacionih regiona'))

estimate = initial_estimate(coordinates,region_number)

print('POcetne nasumicno generisane procene u formatu [ax,ay] :', estimate)

# generate discretized point space within the coordinates
test_points = generate_points(coordinates)
prob_point = find_prob(coordinates,test_points,probabilities)

for i in range(20):
	print('Iteracija ', i)
	# konstruise Voronoi regione na osnovu procene tezista 
	voronoi_kdtree = cKDTree(estimate)
	test_point_dist, test_point_regions = voronoi_kdtree.query(test_points, k=1)

	# test_point_regions sada sadrzi niz oblika (n_test, 1)
	# sa indeksima tacaka u voronoi_points najblizim svakoj od nasih test tacaka.
	test_point_regions = np.asarray([x for x in test_point_regions])
	expectation = np.zeros((region_number,2))
	new_estimate = np.zeros((region_number,2))

	for region_index in range(region_number):		# prolazi kroz svaki region
		denominator = 0
		point_indices = np.where(test_point_regions==region_index)	# trazi tacke koje odgovaraju tom regionu
		for point_index in point_indices[0]:	# iterira te tacke 
			current_index = int(point_index)						# nalazi index tacke na kojoj se nalazimo
			current_prob = prob_point[current_index][0]				# trazi odgovarajucu verovatnocu p(x|Ri)=p(x,Ri)/p(Ri)
			denominator += current_prob
			increment = np.asarray([x * current_prob for x in test_points[current_index]])	# nalazi produkt x*p(x|Ri)
			expectation[region_index] = expectation[region_index] + increment	# azurira ocekivanu vrednost
		expectation[region_index] = expectation[region_index]/denominator
		new_estimate[region_index] = np.around(expectation[region_index],decimals=2) 			#trazi najblizu tacku ocekivane vrednosti
		print('Nova procena za region', region_index,' je : ',new_estimate[region_index] )

	error = np.sum(np.square(new_estimate-estimate))
	vor = Voronoi(estimate)
	voronoi_plot_2d(vor)
	
	plt.show(block=False)
	plt.pause(3)
	if (abs(error)<0.02):
		print('ALGORITAM JE KONVERGIRAO ! TO BRE!')
		break
	plt.close()
	estimate = new_estimate
