def featureScaling(arr):
	min_feature = arr[0]
	max_feature = arr[0]
	for feature in arr:
		if feature > max_feature:
			max_feature = feature
		if feature < min_feature:
			min_feature = feature

	scaled_feature = []
	divider = float(max_feature - min_feature)
	for feature in arr:
		scaled_feature.append((feature - min_feature) / divider)

	return scaled_feature

data = [115, 140, 175]
print featureScaling(data)
