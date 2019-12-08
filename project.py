from PIL import Image
import glob
import numpy as np
import sys
import math
import time
import pandas

padding=20
with_face=499
non_face=2000
# with_face = 100
# non_face=200

def open_image(path):
	image_list = []
	for filename in glob.glob(path+'\\'+'*.png') :
		im = Image.open(filename)
		image_list.append(np.array(im))
	data = np.array(image_list)
	#print(data.shape)
	return data

def compute_image_sum(image_data):
	data_with_padding=np.zeros((image_data.shape[0], 19+padding*2, 19+padding*2))
	#print(image_data.shape, data_with_padding.shape, data_with_padding[:, 0:19, padding:19+padding].shape)
	data_with_padding[:, padding:19+padding, padding:19+padding]=image_data
	image_sum=np.zeros((data_with_padding.shape))
	# image_sum[:, 0, :]=data_with_padding[:, 0, :]
	print("image_sum", image_sum.shape)
	for i in range(image_sum.shape[0]):
		for j in range(1, image_sum.shape[1]):
			row_sum=0.0
			for k in range(image_sum.shape[2]):
				row_sum+=data_with_padding[i][j][k]
				image_sum[i][j][k]=image_sum[i][j-1][k]+row_sum
	return image_sum

def compute_rectangle_pixel_sum(cur_image_sum, topleft, downright):
	#print(downright[0], downright[1], topleft[0], topleft[1], topleft[0], downright[1], downright[0], topleft[1])
	value=cur_image_sum[int(downright[0])][int(downright[1])]+ cur_image_sum[int(topleft[0])][int(topleft[1])] \
			 -(cur_image_sum[int(topleft[0])][int(downright[1])] + cur_image_sum[int(downright[0])][int(topleft[1])])
	return value


def compute_feature_two_horizontal(cur_image_sum, window_size):
	#the up rectangle - the bottom rectangle
	feature=np.zeros((19, 19))
	for i in range(19):
		for j in range(19):
			add_rec_topleft=(i+padding, j+padding-window_size[1]/2)
			add_rec_downright=(i+padding+ window_size[0], j+padding+ window_size[1]/2)
			minus_rec_topleft=(i+padding+window_size[0], j+padding-window_size[1]/2)
			minus_rec_downright=(i+padding+window_size[0]*2, j+padding+window_size[1]/2)
			feature[i][j]=compute_rectangle_pixel_sum(cur_image_sum, add_rec_topleft, add_rec_downright)-\
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec_topleft, minus_rec_downright)
	return feature

def compute_feature_two_vertical(cur_image_sum, window_size):
	#the right rectangle - the left rectangle
	feature=np.zeros((19, 19))
	for i in range(19):
		for j in range(19):
			add_rec_topleft=(i+padding, j+padding-window_size[1])
			add_rec_downright=(i+padding+ window_size[0], j+padding)
			minus_rec_topleft=(i+padding, j+padding)
			minus_rec_downright=(i+padding+window_size[0], j+padding+window_size[1])
			feature[i][j]=compute_rectangle_pixel_sum(cur_image_sum, add_rec_topleft, add_rec_downright)-\
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec_topleft, minus_rec_downright)
	return feature

def compute_feature_three_horizontal(cur_image_sum, window_size):
	# the middle-the up rectangle - the bottom rectangle
	feature=np.zeros((19, 19))
	for i in range(19):
		for j in range(19):
			# the top left of the middle rectangle is in the position i,j
			add_rec_topleft=(i+padding, j+padding)
			add_rec_downright=(i+padding+ window_size[0], j+padding+ window_size[1])
			minus_rec1_topleft=(i+padding-window_size[0], j+padding)
			minus_rec1_downright=(i+padding, j+padding+window_size[1])
			minus_rec2_topleft = (i+padding + window_size[0], j + padding )
			minus_rec2_downright = (i+padding + window_size[0] * 2, j + padding + window_size[1])
			feature[i][j]=compute_rectangle_pixel_sum(cur_image_sum, add_rec_topleft, add_rec_downright)*2 - \
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec1_topleft, minus_rec1_downright)- \
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec2_topleft, minus_rec2_downright)
	return feature

def compute_feature_three_vertical(cur_image_sum, window_size):
	# the middle-the up rectangle - the bottom rectangle
	feature=np.zeros((19, 19))
	for i in range(19):
		for j in range(19):
			# the top left of the middle rectangle is in the position i,j
			add_rec_topleft=(i+padding, j+padding)
			add_rec_downright=(i+padding+ window_size[0], j+padding+ window_size[1])
			minus_rec1_topleft=(i+padding, j+padding-window_size[1])
			minus_rec1_downright=(i+padding+window_size[0], j+padding)
			minus_rec2_topleft = (i+padding , j + padding + window_size[1] )
			minus_rec2_downright = (i+padding + window_size[0] , j + padding + window_size[1]*2)
			feature[i][j]=compute_rectangle_pixel_sum(cur_image_sum, add_rec_topleft, add_rec_downright)*2 - \
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec1_topleft, minus_rec1_downright)- \
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec2_topleft, minus_rec2_downright)
	return feature

def compute_feature_four(cur_image_sum, window_size):
	# top right+ bottom left - top left- bottom right
	feature=np.zeros((19, 19))
	for i in range(19):
		for j in range(19):
			# the middle point of 4 rectangles is in the position i,j
			add_rec1_topleft=(i+padding- window_size[0], j+padding)
			add_rec1_downright=(i+padding, j+padding+ window_size[1])
			add_rec2_topleft = (i + padding , j + padding-window_size[1])
			add_rec2_downright = (i + padding + window_size[0], j + padding )
			minus_rec1_topleft=(i+padding-window_size[0], j+padding-window_size[1])
			minus_rec1_downright=(i+padding, j+padding)
			minus_rec2_topleft = (i+padding , j + padding )
			minus_rec2_downright = (i+padding + window_size[0], j + padding + window_size[1])
			feature[i][j]=compute_rectangle_pixel_sum(cur_image_sum, add_rec1_topleft, add_rec1_downright)+ \
			              compute_rectangle_pixel_sum(cur_image_sum, add_rec2_topleft, add_rec2_downright)- \
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec1_topleft, minus_rec1_downright)- \
			              compute_rectangle_pixel_sum(cur_image_sum, minus_rec2_topleft, minus_rec2_downright)
	return feature


def compute_features(data, window_size_range):
	number_of_sample=data.shape[0]
	feature=np.zeros((number_of_sample, len(window_size_range)*5, 19, 19))
	image_sum=compute_image_sum(data)
	for i in range(len(window_size_range)):
		window_size=window_size_range[i]
		print("now come to window size [%s] = %s" %(i, window_size))
		for n in range(number_of_sample):
			feature[n, i*5, :, :]=compute_feature_two_horizontal(image_sum[n], window_size)
			feature[n, i * 5+1, :, :] =compute_feature_two_vertical(image_sum[n], window_size)
			feature[n, i * 5 + 2, :, :] =compute_feature_three_horizontal(image_sum[n], window_size)
			feature[n, i * 5 + 3, :, :] = compute_feature_three_vertical(image_sum[n], window_size)
			feature[n, i * 5 + 4, :, :] = compute_feature_four(image_sum[n], window_size)
	
	return feature

def data_process():
	train_x_positive = open_image(path=r'dataset\trainset\faces')
	train_x_negative = open_image(path=r'dataset\trainset\non-faces')
	train_y_positive = np.ones((train_x_positive.shape[0], ))
	train_y_negative = np.zeros((train_x_negative.shape[0], ))
	
	train_x = np.concatenate((train_x_positive, train_x_negative), axis=0)
	train_y = np.concatenate((train_y_positive, train_y_negative), axis=0)
	
	print("train data", train_x.shape, train_y.shape)
	
	test_x_positive = open_image(path=r'dataset\testset\faces')
	test_x_negative = open_image(path=r'dataset\testset\non-faces')
	test_y_positive = np.ones((test_x_positive.shape[0], ))
	test_y_negative = np.zeros((test_x_negative.shape[0], ))
	
	test_x = np.concatenate((test_x_positive, test_x_negative), axis=0)
	test_y = np.concatenate((test_y_positive, test_y_negative), axis=0)
	
	print("test data", test_x.shape, test_y.shape)
	
	# window_size_list=[(2,4)]
	window_height = [2,4,6,]
	window_width = [2,6,]
	window_size_list = []
	for h in window_height :
		for w in window_width :
			window_size_list.append((h, w))
	print("window_size_list", len(window_size_list), window_size_list)
	
	train_feature = compute_features(train_x, window_size_list)
	print("train_feature", train_feature.shape)
	
	test_feature=compute_features(test_x, window_size_list)
	print("test_feature", test_feature.shape)
	
	np.save('train_feature.npy', train_feature)
	np.save('test_feature.npy', test_feature)
	np.save('train_x.npy', train_x)
	np.save('train_y.npy', train_y)
	np.save('test_x.npy', test_x)
	np.save('test_y.npy', test_y)
	return 1

def check_festure(data):
	print("check_festure", data.shape)
	s=np.sum(data, axis=2)
	s=np.sum(s, axis=2)
	s=np.sum(s, axis=1)
	print(s)
	return 1

def normalize_weight(weight):
	s=np.sum(weight)
	for i in range(weight.shape[0]):
		weight[i]=weight[i]/s
	return weight

def update_weight(input_weight, prediction, y, beta):
	weight=input_weight.copy()
	for i in range(weight.shape[0]):
		e = 0.0
		if(prediction[i]!=y[i]):
			# print("correction prediction, less weight")
			e=1.0
		# else:
		# 	print("wrong prediction, more weight")
		weight[i]=weight[i]* math.pow(beta, 1.0-e)
	return weight

def find_threshold(feature_array, weight, train_y):
	array_ordered=feature_array.copy()
	array_ordered=np.sort(array_ordered)
	# print("array_ordered", array_ordered)
	best_thre=0.0
	min_error=1.0
	count=0
	for i in range(int(array_ordered.shape[0]/2.0), array_ordered.shape[0]-1):
		if(count>20):
			# print("break for round %s" %i)
			break
		thre=(array_ordered[i]+array_ordered[i+1])/2.0
		error=0.0
		for j in range(weight.shape[0]):
			if((feature_array[j]>=thre and train_y[j]==0) or (feature_array[j]<thre and train_y[j]==1)):
				error+=weight[j]
		# print("find_threshold for i= %s, error=%s, min_error=%s" %(i, error, min_error))
		if(error<min_error):
			min_error=error
			best_thre=thre
			count=0
		else:
			count+=1
	count=0
	index_range=list(range(int(array_ordered.shape[0]/2.0)))
	index_range.reverse()
	for i in index_range:
		if(count>20): break
		thre=(array_ordered[i]+array_ordered[i+1])/2.0
		error=0.0
		for j in range(weight.shape[0]):
			if ((feature_array[j] >= thre and train_y[j] == 0) or (feature_array[j] < thre and train_y[j] == 1)):
				error+=weight[j]
		if(error<min_error):
			min_error=error
			best_thre=thre
			count=0
		else:
			count+=1
	return best_thre, min_error
	

def find_classifier(train_feature, weight, train_y):
	error_min=1.0
	classifier=[0]*5
	t=time.time()
	for m in range(train_feature.shape[1]):
		for row in range(19):
			for col in range(19):
				feature_arr=train_feature[:, m, row, col].copy()
				# print("feature_arr", feature_arr.shape)
				thre, error=find_threshold(feature_arr, weight, train_y)
				if(error<error_min):
					error_min=error
					classifier=[error_min,thre, m, row, col]
		print("come to map %s, min error is %s, time used %s,  " % (m, error_min, time.time() - t))
		t = time.time()
	return classifier

def validate(classifier_list, feature, test_y):
	number_sample=feature.shape[0]
	prediction=np.zeros((number_sample, ))
	alpha_sum=0.0
	for classifier in classifier_list :
		alpha_sum+=classifier[5]
	true_possitive=0
	true_negative=0
	false_possitive=0
	false_negative=0
	for i in range(number_sample):
		pre=0.0
		for classifier in classifier_list:
			thre=classifier[1]
			m=classifier[2]
			row=classifier[3]
			col=classifier[4]
			alpha=classifier[5]
			if(feature[i][m][row][col]>=thre):
				pre+=alpha
		if(pre>=0.5*alpha_sum):
			prediction[i]=1
		if(prediction[i]==0 and test_y[i]==0): true_negative+=1
		elif(prediction[i]==1 and test_y[i]==1): true_possitive+=1
		elif(prediction[i]==0 and test_y[i]==1): false_negative+=1
		else: false_possitive+=1
	print("prediction", prediction.shape, feature.shape, prediction[:20], prediction[700:720])
	status=[true_possitive, true_negative, false_possitive, false_negative,
	        (true_possitive+true_negative)/(true_possitive+true_negative+false_possitive+false_negative),
	        true_possitive/(true_possitive+false_negative), true_negative/(true_negative+false_possitive)]
	return  status, prediction

def train():
	train_feature=np.load('train_feature.npy')
	test_feature=np.load('test_feature.npy')
	train_x=np.load('train_x.npy')
	train_y=np.load('train_y.npy')
	test_x=np.load('test_x.npy')
	test_y=np.load('test_y.npy')
	
	print("train_feature %s, train_y %s " % (train_feature.shape, train_y.shape), train_feature[:10, 2, 10, 10])
	print("test_feature %s, test_y %s " % (test_feature.shape, test_y.shape))
	# check_festure(train_feature)
	# check_festure(test_feature)
	
	
	
	
	classifier_list=[]
	accuracy_list=[]
	weight = np.zeros((with_face + non_face,))
	for i in range(with_face):
		weight[i]=1/(2.0*with_face)
	for i in range(non_face):
		weight[i+with_face]=1/(2.0* non_face)
	
	T = 10
	for t in range(T):
		weight=normalize_weight(weight)
		print("weight", weight.shape, weight[:10], weight[with_face-5: with_face+5])
		classifier=find_classifier(train_feature=train_feature, weight= weight, train_y=train_y)
		error=classifier[0]
		beta_t=error/(1-error)
		alpha_t=-math.log10(beta_t)
		classifier.append(alpha_t)
		classifier_list.append(classifier)
		print("classifier at time %s is %s (error_min, thre, m, row, col, alpha)" %(t, classifier)) #error,thre, m, row, col, alpha
		print("train feature",train_feature.shape, train_feature[:10, 2, 10, 10])
		train_accuracy, train_prediction = validate(classifier_list, train_feature, train_y)
		print("train accuracy for time %s is %s " % (t, train_accuracy))
		weight = update_weight(input_weight=weight, prediction=train_prediction, beta=beta_t, y=train_y)
		test_accuracy, test_prediction=validate(classifier_list, test_feature, test_y)
		acc=train_accuracy.copy()
		acc.extend(test_accuracy)
		accuracy_list.append(acc)
		# accuracy_list.append(test_accuracy)
		
		print("test accuracy for time %s is %s " %(t, test_accuracy))
		
	classifier_np=np.array(classifier_list)
	accuracy_np=np.array(accuracy_list)
	np.save('classifier.npy', classifier_np)
	np.save('accuracy.npy', accuracy_np)
	np.save('weight.npy', weight)
	classifier_df=pandas.DataFrame(classifier_np, columns=['error_min', 'threshold', 'map', 'row', 'col', 'alpha'])
	accuracy_df=pandas.DataFrame(accuracy_np, columns=['training_true_possitive', 'training_true_negative',
	           'training_false_possitive', 'training_false_negative','training_accuracy', 'training_true_positive_rate',
	           'training_true_negative_rate',  'test_true_possitive', 'test_true_negative', 'test_false_possitive',
	            'test_false_negative','test_accuracy', 'test_true_positive_rate', 'test_true_negative_rate'])
	classifier_df.to_excel('classifier.xlsx', index=False )
	accuracy_df.to_excel('accuracy.xlsx', index=False)
	return 1


data_process()
train()
